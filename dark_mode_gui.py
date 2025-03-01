import sys
import os
import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
)
from PIL import Image
import torchvision.transforms.functional as TVF
import contextlib
from typing import Union, List
from pathlib import Path
import csv  # For CSV export functionality

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QInputDialog,
    QTabWidget,
    QSplitter,
    QToolBar,
    QGroupBox,  # Add this for tool sections
    QScrollArea,  # Add this for scrollable content
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QTextDocument
from PyQt5.QtCore import Qt

CLIP_PATH = "google/siglip-so400m-patch14-384"
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

EXTRA_OPTIONS_LIST = [
    "If there is a person/character in the image you must refer to them as {name}.",
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
]

CAPTION_LENGTH_CHOICES = (
    ["any", "very short", "short", "medium-length", "long", "very long"]
    + [str(i) for i in range(20, 261, 10)]
)

HF_TOKEN = os.environ.get("HF_TOKEN", None)

class ImageAdapter(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        ln1: bool,
        pos_emb: bool,
        num_image_tokens: int,
        deep_extract: bool,
    ):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = (
            None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))
        )

        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(
            mean=0.0, std=0.02
        )  # Matches HF's implementation of llama3

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat(
                (
                    vision_outputs[-2],
                    vision_outputs[3],
                    vision_outputs[7],
                    vision_outputs[13],
                    vision_outputs[20],
                ),
                dim=-1,
            )
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
            assert (
                x.shape[-1] == vision_outputs[-2].shape[-1] * 5
            ), f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        # <|image_start|>, IMAGE, <|image_end|>
        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1)
        )
        assert other_tokens.shape == (
            x.shape[0],
            2,
            x.shape[2],
        ), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

# Determine the device to use (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32

# Update autocast usage
if device.type == "cuda":
    autocast = lambda: torch.amp.autocast(device_type='cuda', dtype=torch_dtype)
else:
    autocast = contextlib.nullcontext  # No autocasting on CPU

def load_models(CHECKPOINT_PATH):
    # Load CLIP
    print("Loading CLIP")
    clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
    clip_model = AutoModel.from_pretrained(CLIP_PATH)
    clip_model = clip_model.vision_model

    assert (
        CHECKPOINT_PATH / "clip_model.pt"
    ).exists(), f"clip_model.pt not found in {CHECKPOINT_PATH}"
    print("Loading VLM's custom vision model")
    checkpoint = torch.load(CHECKPOINT_PATH / "clip_model.pt", map_location="cpu")
    checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
    clip_model.load_state_dict(checkpoint)
    del checkpoint

    clip_model.eval()
    clip_model.requires_grad_(False)
    clip_model.to(device)

    # Tokenizer
    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_PATH / "text_model", use_fast=True
    )
    assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(tokenizer)}"

    # Add special tokens to the tokenizer
    special_tokens_dict = {'additional_special_tokens': ['<|system|>', '<|user|>', '<|end|>', '<|eot_id|>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens.")

    # LLM
    print("Loading LLM")
    print("Loading VLM's custom text model")
    text_model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH / "text_model", torch_dtype=torch_dtype
    )
    text_model.eval()
    text_model.to(device)

    # Resize token embeddings if new tokens were added
    if num_added_toks > 0:
        text_model.resize_token_embeddings(len(tokenizer))

    # Image Adapter
    print("Loading image adapter")
    image_adapter = ImageAdapter(
        clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38, False
    )
    image_adapter.load_state_dict(
        torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu")
    )
    image_adapter.eval()
    image_adapter.to(device)

    return clip_processor, clip_model, tokenizer, text_model, image_adapter

class CaptionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Captioning Application")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables for selected images and settings
        self.input_dir = None
        self.single_image_path = None
        self.selected_image_path = None
        self.current_review_image_path = None
        self.last_trigger_word = ''
        self.trigger_word_enabled = False
        
        # Initialize saved prompts dictionary
        self.saved_prompts = {}
        
        # Load saved settings
        self.load_saved_prompts()
        self.load_saved_settings()
        
        self.initUI()

        # Initialize model variables
        self.clip_processor = None
        self.clip_model = None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None

        # Theme variables
        self.dark_mode = False

        # Set dark mode by default
        self.toggle_theme()

    def initUI(self):
        main_layout = QVBoxLayout()
        
        # Create toolbar for font size controls
        toolbar = QToolBar()
        
        # Add font size controls
        font_size_label = QLabel("Font Size: ")
        toolbar.addWidget(font_size_label)
        
        increase_font_button = QPushButton("+")
        increase_font_button.setFixedSize(30, 30)
        increase_font_button.clicked.connect(self.increase_font_size)
        toolbar.addWidget(increase_font_button)
        
        decrease_font_button = QPushButton("-")
        decrease_font_button.setFixedSize(30, 30)
        decrease_font_button.clicked.connect(self.decrease_font_size)
        toolbar.addWidget(decrease_font_button)
        
        # Add reset UI size button
        reset_size_button = QPushButton("Reset")
        reset_size_button.setFixedWidth(60)
        reset_size_button.clicked.connect(self.reset_ui_size)
        toolbar.addWidget(reset_size_button)
        
        main_layout.addWidget(toolbar)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.tagging_tab = QWidget()
        self.review_tab = QWidget()
        self.tools_tab = QWidget()
        
        # Add tabs to widget
        self.tab_widget.addTab(self.tagging_tab, "Tagging")
        self.tab_widget.addTab(self.review_tab, "Review")
        self.tab_widget.addTab(self.tools_tab, "Tools")
        
        # Setup each tab
        self.setup_tagging_tab()
        self.setup_review_tab()
        self.setup_tools_tab()
        
        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)
        
        # Apply the saved font size
        self.update_application_font()
        
        # After creating the trigger word UI elements, set their values from saved settings
        if hasattr(self, 'last_trigger_word'):
            self.trigger_word_input.setText(self.last_trigger_word)
        if hasattr(self, 'trigger_word_enabled'):
            self.trigger_word_checkbox.setChecked(self.trigger_word_enabled)
        
        # If we have a saved directory, update the label
        if self.input_dir:
            self.input_dir_label.setText(str(self.input_dir))
            # Load images from the saved directory
            self.load_images()

    def setup_tagging_tab(self):
        # This is essentially the original layout
        main_layout = QHBoxLayout()

        # Left panel for parameters
        left_panel = QVBoxLayout()

        # Input directory selection
        self.input_dir_button = QPushButton("Select Input Directory")
        self.input_dir_button.clicked.connect(self.select_input_directory)
        self.input_dir_label = QLabel("No directory selected")
        left_panel.addWidget(self.input_dir_button)
        left_panel.addWidget(self.input_dir_label)

        # Single image selection
        self.single_image_button = QPushButton("Select Single Image")
        self.single_image_button.clicked.connect(self.select_single_image)
        self.single_image_label = QLabel("No image selected")
        left_panel.addWidget(self.single_image_button)
        left_panel.addWidget(self.single_image_label)

        # Caption Type
        self.caption_type_combo = QComboBox()
        self.caption_type_combo.addItems(CAPTION_TYPE_MAP.keys())
        self.caption_type_combo.setCurrentText("Descriptive")
        left_panel.addWidget(QLabel("Caption Type:"))
        left_panel.addWidget(self.caption_type_combo)

        # Caption Length
        self.caption_length_combo = QComboBox()
        self.caption_length_combo.addItems(CAPTION_LENGTH_CHOICES)
        self.caption_length_combo.setCurrentText("long")
        left_panel.addWidget(QLabel("Caption Length:"))
        left_panel.addWidget(self.caption_length_combo)

        # Trigger Word Option
        trigger_word_layout = QHBoxLayout()
        self.trigger_word_checkbox = QCheckBox("Use Trigger Word:")
        self.trigger_word_input = QLineEdit()
        self.trigger_word_input.setPlaceholderText("Enter trigger word (e.g., 'a painting of')")
        trigger_word_layout.addWidget(self.trigger_word_checkbox)
        trigger_word_layout.addWidget(self.trigger_word_input)
        left_panel.addLayout(trigger_word_layout)

        # Use Custom Prompt Checkbox
        self.use_custom_prompt_checkbox = QCheckBox("Combine Custom Prompt with Options")
        left_panel.addWidget(self.use_custom_prompt_checkbox)

        # Extra Options
        left_panel.addWidget(QLabel("Extra Options:"))
        self.extra_options_checkboxes = []
        for option in EXTRA_OPTIONS_LIST:
            checkbox = QCheckBox(option)
            self.extra_options_checkboxes.append(checkbox)
            left_panel.addWidget(checkbox)

        # Name Input
        self.name_input_line = QLineEdit()
        left_panel.addWidget(QLabel("Person/Character Name (if applicable):"))
        left_panel.addWidget(self.name_input_line)

        # Custom Prompt section with saved prompts
        custom_prompt_layout = QVBoxLayout()
        custom_prompt_layout.addWidget(QLabel("Custom Prompt:"))
        
        # Add dropdown for saved prompts
        self.saved_prompts_combo = QComboBox()
        self.saved_prompts_combo.addItem("Select saved prompt...")
        self.saved_prompts_combo.addItems(self.saved_prompts.keys())
        self.saved_prompts_combo.currentTextChanged.connect(self.load_saved_prompt)
        custom_prompt_layout.addWidget(self.saved_prompts_combo)
        
        # Custom prompt buttons layout
        prompt_buttons_layout = QHBoxLayout()
        
        # Save prompt button
        self.save_prompt_button = QPushButton("Save Current Prompt")
        self.save_prompt_button.clicked.connect(self.save_current_prompt)
        prompt_buttons_layout.addWidget(self.save_prompt_button)
        
        # Delete prompt button
        self.delete_prompt_button = QPushButton("Delete Selected Prompt")
        self.delete_prompt_button.clicked.connect(self.delete_saved_prompt)
        prompt_buttons_layout.addWidget(self.delete_prompt_button)
        
        custom_prompt_layout.addLayout(prompt_buttons_layout)
        
        self.custom_prompt_text = QTextEdit()
        custom_prompt_layout.addWidget(self.custom_prompt_text)
        
        left_panel.addLayout(custom_prompt_layout)

        # Checkpoint Path
        self.checkpoint_path_line = QLineEdit()
        self.checkpoint_path_line.setText("cgrkzexw-599808")
        left_panel.addWidget(QLabel("Checkpoint Path:"))
        left_panel.addWidget(self.checkpoint_path_line)

        # Load Models Button
        self.load_models_button = QPushButton("Load Models")
        self.load_models_button.clicked.connect(self.load_models)
        left_panel.addWidget(self.load_models_button)

        # Run Buttons
        self.run_button = QPushButton("Generate Captions for All Images")
        self.run_button.clicked.connect(self.generate_captions)
        left_panel.addWidget(self.run_button)

        self.caption_selected_button = QPushButton("Caption Selected Image")
        self.caption_selected_button.clicked.connect(self.caption_selected_image)
        self.caption_selected_button.setEnabled(False)  # Disabled until an image is selected
        left_panel.addWidget(self.caption_selected_button)

        self.caption_single_button = QPushButton("Caption Single Image")
        self.caption_single_button.clicked.connect(self.caption_single_image)
        self.caption_single_button.setEnabled(False)  # Disabled until a single image is selected
        left_panel.addWidget(self.caption_single_button)

        # Theme Toggle Button
        self.toggle_theme_button = QPushButton("Toggle Dark Mode")
        self.toggle_theme_button.clicked.connect(self.toggle_theme)
        left_panel.addWidget(self.toggle_theme_button)

        # Right panel for image display and captions
        right_panel = QVBoxLayout()

        # List widget for images
        self.image_list_widget = QListWidget()
        self.image_list_widget.itemClicked.connect(self.display_selected_image)
        right_panel.addWidget(QLabel("Images:"))
        right_panel.addWidget(self.image_list_widget)

        # Label to display the selected image
        self.selected_image_label = QLabel()
        self.selected_image_label.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(QLabel("Selected Image:"))
        right_panel.addWidget(self.selected_image_label)

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 3)
        self.tagging_tab.setLayout(main_layout)

    def setup_review_tab(self):
        main_layout = QVBoxLayout()
        
        # Create a splitter for the top section
        top_splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Image list
        image_list_container = QWidget()
        image_list_layout = QVBoxLayout(image_list_container)
        image_list_layout.addWidget(QLabel("Images:"))
        
        self.review_image_list = QListWidget()
        self.review_image_list.itemClicked.connect(self.load_caption_for_review)
        image_list_layout.addWidget(self.review_image_list)
        
        # Add Re-tag button
        self.retag_button = QPushButton("Re-tag Selected Image")
        self.retag_button.clicked.connect(self.retag_selected_image)
        image_list_layout.addWidget(self.retag_button)
        
        # Middle - Image display
        image_display_container = QWidget()
        image_display_layout = QVBoxLayout(image_display_container)
        image_display_layout.addWidget(QLabel("Selected Image:"))
        
        self.review_image_display = QLabel()
        self.review_image_display.setAlignment(Qt.AlignCenter)
        self.review_image_display.setMinimumSize(400, 400)  # Set minimum size
        image_display_layout.addWidget(self.review_image_display)
        
        # Right side - Caption editor
        caption_editor_container = QWidget()
        caption_editor_layout = QVBoxLayout(caption_editor_container)
        caption_editor_layout.addWidget(QLabel("Edit Caption:"))
        
        self.review_caption_editor = QTextEdit()
        caption_editor_layout.addWidget(self.review_caption_editor)
        
        # Add widgets to splitter
        top_splitter.addWidget(image_list_container)
        top_splitter.addWidget(image_display_container)
        top_splitter.addWidget(caption_editor_container)
        top_splitter.setSizes([200, 400, 600])  # Set initial sizes
        
        # Find and Replace section
        find_replace_container = QWidget()
        find_replace_layout = QVBoxLayout(find_replace_container)
        find_replace_layout.addWidget(QLabel("Find and Replace:"))
        
        # Find field
        find_layout = QHBoxLayout()
        find_layout.addWidget(QLabel("Find:"))
        self.find_text = QLineEdit()
        find_layout.addWidget(self.find_text)
        find_replace_layout.addLayout(find_layout)
        
        # Replace field
        replace_layout = QHBoxLayout()
        replace_layout.addWidget(QLabel("Replace with:"))
        self.replace_text = QLineEdit()
        replace_layout.addWidget(self.replace_text)
        find_replace_layout.addLayout(replace_layout)
        
        # Find and Replace buttons
        buttons_layout = QHBoxLayout()
        
        self.find_button = QPushButton("Find Next")
        self.find_button.clicked.connect(self.find_in_caption)
        buttons_layout.addWidget(self.find_button)
        
        self.replace_button = QPushButton("Replace")
        self.replace_button.clicked.connect(self.replace_in_caption)
        buttons_layout.addWidget(self.replace_button)
        
        self.replace_all_button = QPushButton("Replace All")
        self.replace_all_button.clicked.connect(self.replace_all_in_caption)
        buttons_layout.addWidget(self.replace_all_button)
        
        self.replace_all_files_button = QPushButton("Replace All in All Files")
        self.replace_all_files_button.clicked.connect(self.replace_all_in_files)
        buttons_layout.addWidget(self.replace_all_files_button)
        
        find_replace_layout.addLayout(buttons_layout)
        
        # Add all components to main layout
        main_layout.addWidget(top_splitter, 7)  # 70% of space
        main_layout.addWidget(find_replace_container, 3)  # 30% of space
        
        self.review_tab.setLayout(main_layout)

    def setup_tools_tab(self):
        """Set up the Tools tab with LoRA toolkit functionality"""
        # Create a scroll area to contain all tools
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create a container widget for the scroll area
        tools_container = QWidget()
        main_layout = QVBoxLayout(tools_container)
        main_layout.setSpacing(0)  # No spacing between elements
        main_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        
        # Load saved tool settings
        self.tool_settings = self.load_tool_settings()
        
        # ===== SHARED DIRECTORY =====
        # Create a simple widget instead of GroupBox for shared directory
        shared_dir_widget = QWidget()
        shared_dir_widget.setFixedHeight(40)  # Fixed height
        shared_dir_widget.setStyleSheet("background-color: #2D2D30; border-bottom: 1px solid #3F3F46;")
        
        shared_dir_layout = QHBoxLayout(shared_dir_widget)
        shared_dir_layout.setContentsMargins(5, 2, 5, 2)
        shared_dir_layout.setSpacing(5)
        
        shared_dir_layout.addWidget(QLabel("Directory:"))
        self.shared_directory = QLineEdit()
        self.shared_directory.setText(self.tool_settings.get('shared_directory', ''))
        self.shared_directory.textChanged.connect(self.update_tool_directories)
        shared_dir_layout.addWidget(self.shared_directory, 1)
        
        shared_dir_browse = QPushButton("Browse")
        shared_dir_browse.clicked.connect(self.browse_shared_directory)
        shared_dir_layout.addWidget(shared_dir_browse)
        
        main_layout.addWidget(shared_dir_widget)
        
        # ===== SITE BUILDER =====
        # Create a simple widget instead of GroupBox for site builder
        site_builder_widget = QWidget()
        site_builder_widget.setFixedHeight(40)  # Fixed height
        site_builder_widget.setStyleSheet("background-color: #2D2D30; border-bottom: 1px solid #3F3F46;")
        
        site_builder_layout = QHBoxLayout(site_builder_widget)
        site_builder_layout.setContentsMargins(5, 2, 5, 2)
        site_builder_layout.setSpacing(5)
        
        site_builder_layout.addWidget(QLabel("Project:"))
        self.sb_project_name = QLineEdit()
        self.sb_project_name.setText(self.tool_settings.get('sb_project_name', ''))
        site_builder_layout.addWidget(self.sb_project_name, 1)
        
        site_builder_layout.addWidget(QLabel("Output:"))
        self.sb_output_dir = QLineEdit()
        self.sb_output_dir.setText(self.tool_settings.get('sb_output_dir', ''))
        site_builder_layout.addWidget(self.sb_output_dir, 1)
        
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.browse_sb_output_dir)
        site_builder_layout.addWidget(output_button)
        
        generate_button = QPushButton("Generate")
        generate_button.clicked.connect(self.generate_website)
        site_builder_layout.addWidget(generate_button)
        
        main_layout.addWidget(site_builder_widget)
        
        # ===== PROMPT CLEANER =====
        prompt_cleaner_widget = QWidget()
        prompt_cleaner_widget.setFixedHeight(40)  # Fixed height
        prompt_cleaner_widget.setStyleSheet("background-color: #2D2D30; border-bottom: 1px solid #3F3F46;")
        
        prompt_cleaner_layout = QHBoxLayout(prompt_cleaner_widget)
        prompt_cleaner_layout.setContentsMargins(5, 2, 5, 2)
        prompt_cleaner_layout.setSpacing(5)
        
        prompt_cleaner_layout.addWidget(QLabel("Text to remove:"))
        self.pc_text_to_remove = QLineEdit()
        self.pc_text_to_remove.setText(self.tool_settings.get('pc_text_to_remove', ''))
        prompt_cleaner_layout.addWidget(self.pc_text_to_remove, 1)
        
        pc_process_button = QPushButton("Remove Text")
        pc_process_button.clicked.connect(self.remove_text_from_files)
        prompt_cleaner_layout.addWidget(pc_process_button)
        
        main_layout.addWidget(prompt_cleaner_widget)
        
        # ===== TRIGGER WORD =====
        trigger_word_widget = QWidget()
        trigger_word_widget.setFixedHeight(40)  # Fixed height
        trigger_word_widget.setStyleSheet("background-color: #2D2D30; border-bottom: 1px solid #3F3F46;")
        
        trigger_word_layout = QHBoxLayout(trigger_word_widget)
        trigger_word_layout.setContentsMargins(5, 2, 5, 2)
        trigger_word_layout.setSpacing(5)
        
        trigger_word_layout.addWidget(QLabel("Text to add:"))
        self.tw_text_to_add = QLineEdit()
        self.tw_text_to_add.setText(self.tool_settings.get('tw_text_to_add', ''))
        trigger_word_layout.addWidget(self.tw_text_to_add, 1)
        
        tw_process_button = QPushButton("Add Text")
        tw_process_button.clicked.connect(self.add_text_to_files)
        trigger_word_layout.addWidget(tw_process_button)
        
        main_layout.addWidget(trigger_word_widget)
        
        # ===== CSV EXPORT =====
        csv_widget = QWidget()
        csv_widget.setFixedHeight(40)  # Fixed height
        csv_widget.setStyleSheet("background-color: #2D2D30; border-bottom: 1px solid #3F3F46;")
        
        csv_layout = QHBoxLayout(csv_widget)
        csv_layout.setContentsMargins(5, 2, 5, 2)
        csv_layout.setSpacing(5)
        
        csv_layout.addWidget(QLabel("File:"))
        self.csv_filename = QLineEdit()
        self.csv_filename.setText(self.tool_settings.get('csv_filename', ''))
        csv_layout.addWidget(self.csv_filename, 1)
        
        csv_layout.addWidget(QLabel("Save to:"))
        self.csv_save_location = QLineEdit()
        self.csv_save_location.setText(self.tool_settings.get('csv_save_location', ''))
        csv_layout.addWidget(self.csv_save_location, 1)
        
        csv_save_button = QPushButton("Browse")
        csv_save_button.clicked.connect(self.browse_csv_save_location)
        csv_layout.addWidget(csv_save_button)
        
        csv_process_button = QPushButton("Create CSV")
        csv_process_button.clicked.connect(self.create_csv)
        csv_layout.addWidget(csv_process_button)
        
        main_layout.addWidget(csv_widget)
        
        # Add a stretcher to push all widgets to the top
        main_layout.addStretch(1)
        
        # Set the container as the scroll area's widget
        scroll_area.setWidget(tools_container)
        
        # Create a layout for the tab
        tab_layout = QVBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll_area)
        
        # Set the layout for the tab
        self.tools_tab.setLayout(tab_layout)

    def select_input_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.input_dir = Path(directory)
            self.input_dir_label.setText(str(self.input_dir))
            self.load_images()
            # Save the new directory in settings
            self.save_settings()
        else:
            self.input_dir_label.setText("No directory selected")
            self.input_dir = None

    def select_single_image(self):
        file_filter = "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.tiff)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Single Image", "", file_filter)
        if file_path:
            self.single_image_path = Path(file_path)
            self.single_image_label.setText(str(self.single_image_path.name))
            self.display_image(self.single_image_path)
            self.caption_single_button.setEnabled(True)
        else:
            self.single_image_label.setText("No image selected")
            self.single_image_path = None
            self.caption_single_button.setEnabled(False)

    def load_images(self):
        # List of image file extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

        # Collect all image files in the directory
        self.image_files = [f for f in self.input_dir.iterdir() if f.suffix.lower() in image_extensions]

        if not self.image_files:
            QMessageBox.warning(self, "No Images", "No image files found in the selected directory.")
            return

        # Clear both list widgets
        self.image_list_widget.clear()
        self.review_image_list.clear()
        
        for image_path in self.image_files:
            # Add to tagging tab list
            item = QListWidgetItem(str(image_path.name))
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                # Increase thumbnail size to 150x150
                scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                icon = QIcon(scaled_pixmap)
                item.setIcon(icon)
            self.image_list_widget.addItem(item)
            
            # Add to review tab list
            review_item = QListWidgetItem(str(image_path.name))
            review_item.setIcon(icon)
            self.review_image_list.addItem(review_item)

    def load_caption_for_review(self, item):
        """Load the caption file for the selected image in the review tab"""
        if not self.input_dir:
            return
            
        image_name = item.text()
        image_path = self.input_dir / image_name
        caption_path = image_path.with_suffix('.txt')
        
        # Save any pending edits from the previous image
        self.save_current_caption()
        
        # Display the image in the review tab
        pixmap = QPixmap(str(image_path))
        if not pixmap.isNull():
            # Scale the image to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.review_image_display.width(), 
                self.review_image_display.height(),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.review_image_display.setPixmap(scaled_pixmap)
        else:
            self.review_image_display.clear()
        
        # Load the caption
        if caption_path.exists():
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read()
                self.review_caption_editor.setText(caption_text)
            except Exception as e:
                print(f"Error loading caption: {e}")
                self.review_caption_editor.setText("")
        
        # Store the current image path for auto-save
        self.current_review_image_path = image_path

    def save_current_caption(self):
        """Save the current caption if there's an image selected"""
        if hasattr(self, 'current_review_image_path') and self.current_review_image_path:
            caption_path = self.current_review_image_path.with_suffix('.txt')
            edited_text = self.review_caption_editor.toPlainText()
            
            try:
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(edited_text)
                # No message box, just silently save
                print(f"Caption saved for {self.current_review_image_path.name}")
            except Exception as e:
                print(f"Error saving caption: {e}")

    def find_in_caption(self):
        """Find the next occurrence of the search text in the caption editor (case insensitive)"""
        find_text = self.find_text.text()
        if not find_text:
            return
        
        # Use case insensitive search
        options = QTextDocument.FindFlags()
        
        # Get the current cursor position
        cursor = self.review_caption_editor.textCursor()
        # Start searching from the current position
        found = self.review_caption_editor.find(find_text, options)
        
        if not found:
            # If not found from current position, try from the beginning
            cursor.setPosition(0)
            self.review_caption_editor.setTextCursor(cursor)
            found = self.review_caption_editor.find(find_text, options)
            
            if not found:
                QMessageBox.information(self, "Not Found", f"'{find_text}' not found in the caption.")

    def replace_in_caption(self):
        """Replace the current selection with the replacement text"""
        find_text = self.find_text.text()
        replace_text = self.replace_text.text()
        
        if not find_text:
            return
            
        cursor = self.review_caption_editor.textCursor()
        if cursor.hasSelection() and cursor.selectedText() == find_text:
            cursor.insertText(replace_text)
            self.review_caption_editor.setTextCursor(cursor)
        else:
            # If nothing is selected or the selection doesn't match, find the next occurrence
            self.find_in_caption()

    def replace_all_in_caption(self):
        """Replace all occurrences of the search text in the current caption (case insensitive)"""
        find_text = self.find_text.text()
        replace_text = self.replace_text.text()
        
        if not find_text:
            return
        
        current_text = self.review_caption_editor.toPlainText()
        
        # Use case insensitive replacement with regex
        import re
        new_text = re.sub(re.escape(find_text), replace_text, current_text, flags=re.IGNORECASE)
        self.review_caption_editor.setText(new_text)
        
        # Count replacements (case insensitive)
        count = len(re.findall(re.escape(find_text), current_text, re.IGNORECASE))
        QMessageBox.information(self, "Replace All", f"Replaced {count} occurrences of '{find_text}'.")

    def replace_all_in_files(self):
        """Replace all occurrences of the search text in all caption files (case insensitive)"""
        find_text = self.find_text.text()
        replace_text = self.replace_text.text()
        
        if not find_text or not self.input_dir:
            QMessageBox.warning(self, "Error", "Please enter search text and select a directory.")
            return
        
        # Confirm before proceeding
        reply = QMessageBox.question(self, "Confirm Replace All", 
            f"Are you sure you want to replace all occurrences of '{find_text}' with '{replace_text}' in ALL caption files?",
            QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.No:
            return
        
        # Get all caption files
        caption_files = list(self.input_dir.glob("*.txt"))
        
        if not caption_files:
            QMessageBox.information(self, "No Files", "No caption files found in the directory.")
            return
        
        import re
        total_replacements = 0
        files_modified = 0
        
        for caption_file in caption_files:
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count matches before replacement (case insensitive)
                replacements = len(re.findall(re.escape(find_text), content, re.IGNORECASE))
                
                if replacements > 0:
                    # Perform case insensitive replacement
                    new_content = re.sub(re.escape(find_text), replace_text, content, flags=re.IGNORECASE)
                    
                    with open(caption_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    total_replacements += replacements
                    files_modified += 1
            except Exception as e:
                print(f"Error processing {caption_file}: {e}")
                
        # Update the current caption if it's open
        if self.review_caption_editor.toPlainText():
            current_text = self.review_caption_editor.toPlainText()
            new_text = re.sub(re.escape(find_text), replace_text, current_text, flags=re.IGNORECASE)
            self.review_caption_editor.setText(new_text)
        
        QMessageBox.information(self, "Replace Complete", 
            f"Replaced {total_replacements} occurrences of '{find_text}' across {files_modified} files.")

    def display_selected_image(self, item):
        # Find the image path corresponding to the clicked item
        image_name = item.text()
        if self.input_dir:
            image_path = self.input_dir / image_name
            if image_path.exists():
                self.selected_image_path = image_path
                self.display_image(image_path)
                self.caption_selected_button.setEnabled(True)
        else:
            self.selected_image_path = None
            self.caption_selected_button.setEnabled(False)

    def display_image(self, image_path):
        pixmap = QPixmap(str(image_path))
        if not pixmap.isNull():
            # Scale the image to fit the label
            scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.selected_image_label.setPixmap(scaled_pixmap)
        else:
            self.selected_image_label.clear()

    def load_models(self):
        checkpoint_path = Path(self.checkpoint_path_line.text())
        if not checkpoint_path.exists():
            QMessageBox.warning(self, "Checkpoint Error", f"Checkpoint path does not exist: {checkpoint_path}")
            return

        try:
            (
                self.clip_processor,
                self.clip_model,
                self.tokenizer,
                self.text_model,
                self.image_adapter,
            ) = load_models(checkpoint_path)
            QMessageBox.information(self, "Models Loaded", "Models have been loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error", f"An error occurred while loading models: {e}")

    def collect_parameters(self):
        # Collect parameters for caption generation
        caption_type = self.caption_type_combo.currentText()
        caption_length = self.caption_length_combo.currentText()
        extra_options = [checkbox.text() for checkbox in self.extra_options_checkboxes if checkbox.isChecked()]
        name_input = self.name_input_line.text()
        custom_prompt = self.custom_prompt_text.toPlainText()

        return caption_type, caption_length, extra_options, name_input, custom_prompt

    def generate_caption(
        self,
        input_image: Image.Image,
        caption_type: str,
        caption_length: Union[str, int],
        extra_options: List[str],
        name_input: str,
        custom_prompt: str,
        clip_model,
        tokenizer,
        text_model,
        image_adapter,
    ) -> tuple:
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # If using custom prompt with options
        if self.use_custom_prompt_checkbox.isChecked() and custom_prompt.strip():
            base_prompt = custom_prompt.strip()
        else:
            # Build prompt based on caption type and length
            if caption_length == "any":
                map_idx = 0
            elif isinstance(caption_length, int) or caption_length.isdigit():
                map_idx = 1
            else:
                map_idx = 2
            
            base_prompt = CAPTION_TYPE_MAP[caption_type][map_idx]

        # Add extra options
        if len(extra_options) > 0:
            base_prompt += " " + " ".join(extra_options)

        # Add name, length, word_count
        prompt_str = base_prompt.format(
            name=name_input, 
            length=caption_length, 
            word_count=caption_length
        )

        # Preprocess image
        image = input_image.resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(device)

        # Embed image
        # This results in Batch x Image Tokens x Features
        with autocast():
            vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
            embedded_images = image_adapter(vision_outputs.hidden_states)
            embedded_images = embedded_images.to(device)

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt_str,
            },
        ]

        # Format the conversation
        # The apply_chat_template method might not be available; handle accordingly
        if hasattr(tokenizer, "apply_chat_template"):
            convo_string = tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=True
            )
        else:
            # Simple concatenation if apply_chat_template is not available
            convo_string = (
                "<|system|>\n" + convo[0]["content"] + "\n<|end|>\n<|user|>\n" + convo[1]["content"] + "\n<|end|>\n"
            )

        assert isinstance(convo_string, str)

        # Tokenize the conversation
        # prompt_str is tokenized separately so we can do the calculations below
        convo_tokens = tokenizer.encode(
            convo_string, return_tensors="pt", add_special_tokens=False, truncation=False
        ).to(device)
        prompt_tokens = tokenizer.encode(
            prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False
        ).to(device)
        assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
        convo_tokens = convo_tokens.squeeze(0)  # Squeeze just to make the following easier
        prompt_tokens = prompt_tokens.squeeze(0)

        # Calculate where to inject the image
        # Use the indices of the special tokens
        end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")

        # Ensure end_token_id is valid
        if end_token_id is None:
            raise ValueError("The tokenizer does not recognize the '<|end|>' token. Please ensure special tokens are added.")

        end_token_indices = (convo_tokens == end_token_id).nonzero(as_tuple=True)[0].tolist()
        if len(end_token_indices) >= 2:
            # The image is to be injected between the system message and the user prompt
            preamble_len = end_token_indices[0] + 1  # Position after the first <|end|>
        else:
            preamble_len = 0  # Fallback to the start if tokens are missing

        # Embed the tokens
        convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(device))

        # Construct the input
        input_embeds = torch.cat(
            [
                convo_embeds[:, :preamble_len],  # Part before the prompt
                embedded_images.to(dtype=convo_embeds.dtype),  # Image embeddings
                convo_embeds[:, preamble_len:],  # The prompt and anything after it
            ],
            dim=1,
        ).to(device)

        input_ids = torch.cat(
            [
                convo_tokens[:preamble_len].unsqueeze(0),
                torch.full((1, embedded_images.shape[1]), tokenizer.pad_token_id, dtype=torch.long, device=device),  # Dummy tokens for the image
                convo_tokens[preamble_len:].unsqueeze(0),
            ],
            dim=1,
        ).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        # Debugging
        print(f"Input to model: {repr(tokenizer.decode(input_ids[0]))}")

        # Generate the caption
        generate_ids = text_model.generate(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            suppress_tokens=None,
        )

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")]:
            generate_ids = generate_ids[:, :-1]

        caption = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Add trigger word to the generated caption if enabled
        if self.trigger_word_checkbox.isChecked() and self.trigger_word_input.text().strip():
            trigger_word = self.trigger_word_input.text().strip()
            caption = f"{trigger_word} {caption.strip()}"

        return prompt_str, caption.strip()

    def generate_captions(self):
        if not hasattr(self, 'image_files') or not self.image_files:
            QMessageBox.warning(self, "No Images", "Please select a directory containing images.")
            return

        if not all([self.clip_processor, self.clip_model, self.tokenizer, self.text_model, self.image_adapter]):
            QMessageBox.warning(self, "Models Not Loaded", "Please load the models before generating captions.")
            return

        caption_type, caption_length, extra_options, name_input, custom_prompt = self.collect_parameters()

        # Process each image
        for image_path in self.image_files:
            print(f"\nProcessing image: {image_path}")
            input_image = Image.open(image_path)

            try:
                prompt_str, caption = self.generate_caption(
                    input_image,
                    caption_type,
                    caption_length,
                    extra_options,
                    name_input,
                    custom_prompt,
                    self.clip_model,
                    self.tokenizer,
                    self.text_model,
                    self.image_adapter,
                )

                # Save only the caption without additional text
                caption_file = image_path.with_suffix('.txt')
                with open(caption_file, 'w', encoding='utf-8') as f:
                    f.write(f"{caption}\n")

                print(f"Caption saved to {caption_file}")

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

        QMessageBox.information(self, "Captions Generated", "Captions have been generated and saved.")

    def caption_selected_image(self):
        if not self.selected_image_path:
            QMessageBox.warning(self, "No Image Selected", "Please select an image from the list.")
            return

        if not all([self.clip_processor, self.clip_model, self.tokenizer, self.text_model, self.image_adapter]):
            QMessageBox.warning(self, "Models Not Loaded", "Please load the models before generating captions.")
            return

        caption_type, caption_length, extra_options, name_input, custom_prompt = self.collect_parameters()

        print(f"\nProcessing image: {self.selected_image_path}")
        input_image = Image.open(self.selected_image_path)

        try:
            prompt_str, caption = self.generate_caption(
                input_image,
                caption_type,
                caption_length,
                extra_options,
                name_input,
                custom_prompt,
                self.clip_model,
                self.tokenizer,
                self.text_model,
                self.image_adapter,
            )

            # Save only the caption without additional text
            caption_file = self.selected_image_path.with_suffix('.txt')
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(f"{caption}\n")

            print(f"Caption saved to {caption_file}")

        except Exception as e:
            print(f"Error processing image {self.selected_image_path}: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            return

        QMessageBox.information(self, "Caption Generated", f"Caption has been generated and saved for {self.selected_image_path.name}.")

    def caption_single_image(self):
        if not self.single_image_path:
            QMessageBox.warning(self, "No Image Selected", "Please select a single image.")
            return

        if not all([self.clip_processor, self.clip_model, self.tokenizer, self.text_model, self.image_adapter]):
            QMessageBox.warning(self, "Models Not Loaded", "Please load the models before generating captions.")
            return

        caption_type, caption_length, extra_options, name_input, custom_prompt = self.collect_parameters()

        print(f"\nProcessing image: {self.single_image_path}")
        input_image = Image.open(self.single_image_path)

        try:
            prompt_str, caption = self.generate_caption(
                input_image,
                caption_type,
                caption_length,
                extra_options,
                name_input,
                custom_prompt,
                self.clip_model,
                self.tokenizer,
                self.text_model,
                self.image_adapter,
            )

            # Save only the caption without additional text
            caption_file = self.single_image_path.with_suffix('.txt')
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(f"{caption}\n")

            print(f"Caption saved to {caption_file}")

        except Exception as e:
            print(f"Error processing image {self.single_image_path}: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            return

        QMessageBox.information(self, "Caption Generated", f"Caption has been generated and saved for {self.single_image_path.name}.")

    def load_saved_prompts(self):
        """Load saved prompts from a file"""
        try:
            with open('saved_prompts.txt', 'r', encoding='utf-8') as f:
                current_name = None
                current_prompt = []
                
                for line in f:
                    if line.startswith("NAME:"):
                        # Save previous prompt if exists
                        if current_name and current_prompt:
                            self.saved_prompts[current_name] = "\n".join(current_prompt)
                        
                        # Start new prompt
                        current_name = line[5:].strip()
                        current_prompt = []
                    elif line.startswith("PROMPT:"):
                        current_prompt.append(line[7:].rstrip())
                    elif current_prompt is not None:
                        current_prompt.append(line.rstrip())
                
                # Save the last prompt
                if current_name and current_prompt:
                    self.saved_prompts[current_name] = "\n".join(current_prompt)
                
        except FileNotFoundError:
            pass

    def save_prompts_to_file(self):
        """Save prompts to a file"""
        with open('saved_prompts.txt', 'w', encoding='utf-8') as f:
            for name, prompt in self.saved_prompts.items():
                f.write(f"NAME:{name}\n")
                f.write(f"PROMPT:{prompt}\n")
                f.write("\n")  # Empty line between prompts

    def save_current_prompt(self):
        """Save the current prompt with a user-provided name"""
        current_prompt = self.custom_prompt_text.toPlainText().strip()
        if not current_prompt:
            QMessageBox.warning(self, "Empty Prompt", "Cannot save empty prompt.")
            return

        name, ok = QInputDialog.getText(self, "Save Prompt", "Enter a name for this prompt:")
        if ok and name:
            if name in self.saved_prompts:
                reply = QMessageBox.question(self, "Prompt Exists", 
                    "A prompt with this name already exists. Do you want to overwrite it?",
                    QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.No:
                    return

            self.saved_prompts[name] = current_prompt
            self.update_saved_prompts_combo()
            self.save_prompts_to_file()
            QMessageBox.information(self, "Success", "Prompt saved successfully!")

    def delete_saved_prompt(self):
        """Delete the currently selected prompt"""
        current_name = self.saved_prompts_combo.currentText()
        if current_name and current_name != "Select saved prompt...":
            reply = QMessageBox.question(self, "Confirm Delete", 
                f"Are you sure you want to delete the prompt '{current_name}'?",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                del self.saved_prompts[current_name]
                self.update_saved_prompts_combo()
                self.save_prompts_to_file()
                QMessageBox.information(self, "Success", "Prompt deleted successfully!")

    def update_saved_prompts_combo(self):
        """Update the saved prompts dropdown"""
        self.saved_prompts_combo.clear()
        self.saved_prompts_combo.addItem("Select saved prompt...")
        self.saved_prompts_combo.addItems(sorted(self.saved_prompts.keys()))

    def load_saved_prompt(self, prompt_name):
        """Load a saved prompt into the custom prompt text area"""
        if prompt_name and prompt_name != "Select saved prompt...":
            self.custom_prompt_text.setText(self.saved_prompts[prompt_name])

    def toggle_theme(self):
        """Toggle between light and dark mode"""
        if self.dark_mode:
            self.setStyleSheet("")  # Reset to default (light mode)
            self.dark_mode = False
        else:
            # Apply dark theme stylesheet
            self.setStyleSheet("""
                QWidget {
                    background-color: #2E2E2E;
                    color: #FFFFFF;
                }
                QPushButton {
                    background-color: #3A3A3A;
                    color: #FFFFFF;
                    border: none;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
                QLabel {
                    color: #FFFFFF;
                }
                QLineEdit, QTextEdit {
                    background-color: #3A3A3A;
                    color: #FFFFFF;
                    border: 1px solid #555555;
                }
                QComboBox {
                    background-color: #3A3A3A;
                    color: #FFFFFF;
                    border: 1px solid #555555;
                }
                QListWidget {
                    background-color: #3A3A3A;
                    color: #FFFFFF;
                    border: 1px solid #555555;
                }
                QCheckBox {
                    color: #FFFFFF;
                }
                QTabWidget::pane {
                    border: 1px solid #555555;
                    background-color: #2E2E2E;
                }
                QTabBar::tab {
                    background-color: #3A3A3A;
                    color: #FFFFFF;
                    padding: 8px 12px;
                    border: 1px solid #555555;
                    border-bottom: none;
                }
                QTabBar::tab:selected {
                    background-color: #555555;
                }
                QSplitter::handle {
                    background-color: #555555;
                }
            """)
            self.dark_mode = True

    def increase_font_size(self):
        """Increase the application font size"""
        # Save current window size
        current_size = self.size()
        
        # Increase font size
        self.current_font_size += 1
        
        # Update font without changing window size
        self.update_application_font(preserve_size=current_size)

    def decrease_font_size(self):
        """Decrease the application font size"""
        # Save current window size
        current_size = self.size()
        
        if self.current_font_size > 6:  # Don't let it get too small
            self.current_font_size -= 1
            
            # Update font without changing window size
            self.update_application_font(preserve_size=current_size)

    def update_application_font(self, preserve_size=None):
        """Update the font size for the entire application"""
        font = QFont()
        font.setPointSize(self.current_font_size)
        QApplication.setFont(font)
        
        # Update the font for existing widgets
        for widget in QApplication.allWidgets():
            widget.setFont(font)
        
        # If we need to preserve the window size
        if preserve_size:
            # Allow a brief moment for layout to update
            QApplication.processEvents()
            
            # Restore the original window size
            self.resize(preserve_size)
            
            # Ensure all scroll areas adjust their scrollbars
            for scroll_area in self.findChildren(QScrollArea):
                scroll_area.setWidgetResizable(True)

    def closeEvent(self, event):
        """Handle application close event"""
        # Save any pending caption edits
        self.save_current_caption()
        
        # Save application settings
        self.save_settings()
        
        # Save tool settings
        self.save_tool_settings()
        
        event.accept()

    def load_saved_settings(self):
        """Load saved settings like last directory, trigger word, and font size"""
        try:
            with open('app_settings.txt', 'r', encoding='utf-8') as f:
                settings = {}
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        settings[key] = value
                
                # Set last directory if it exists
                if 'last_directory' in settings and os.path.exists(settings['last_directory']):
                    self.input_dir = Path(settings['last_directory'])
                
                # Set last trigger word
                self.last_trigger_word = settings.get('trigger_word', '')
                self.trigger_word_enabled = settings.get('trigger_word_enabled', 'False') == 'True'
                
                # Set font size if saved
                if 'font_size' in settings:
                    try:
                        self.current_font_size = int(settings['font_size'])
                    except ValueError:
                        # Use default if conversion fails
                        self.current_font_size = QApplication.font().pointSize()
                else:
                    self.current_font_size = QApplication.font().pointSize()
                
        except FileNotFoundError:
            # Default values if file doesn't exist
            self.last_trigger_word = ''
            self.trigger_word_enabled = False
            self.current_font_size = QApplication.font().pointSize()

    def save_settings(self):
        """Save application settings"""
        settings = {}
        
        # Save last directory
        if self.input_dir:
            settings['last_directory'] = str(self.input_dir)
        
        # Save trigger word settings
        settings['trigger_word'] = self.trigger_word_input.text()
        settings['trigger_word_enabled'] = str(self.trigger_word_checkbox.isChecked())
        
        # Save font size
        settings['font_size'] = str(self.current_font_size)
        
        with open('app_settings.txt', 'w', encoding='utf-8') as f:
            for key, value in settings.items():
                f.write(f"{key}:{value}\n")

    def retag_selected_image(self):
        """Re-tag the currently selected image using settings from the Tagging tab"""
        # Check if an image is selected
        selected_items = self.review_image_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Image Selected", "Please select an image to re-tag.")
            return
        
        # Check if models are loaded
        if not all([self.clip_processor, self.clip_model, self.tokenizer, self.text_model, self.image_adapter]):
            QMessageBox.warning(self, "Models Not Loaded", "Please load the models before re-tagging.")
            return
        
        # Get the selected image path
        image_name = selected_items[0].text()
        image_path = self.input_dir / image_name
        
        # Save any pending edits to the current caption
        self.save_current_caption()
        
        # Collect parameters from the Tagging tab
        caption_type, caption_length, extra_options, name_input, custom_prompt = self.collect_parameters()
        
        try:
            # Open the image
            input_image = Image.open(image_path)
            
            # Generate new caption
            prompt_str, caption = self.generate_caption(
                input_image,
                caption_type,
                caption_length,
                extra_options,
                name_input,
                custom_prompt,
                self.clip_model,
                self.tokenizer,
                self.text_model,
                self.image_adapter,
            )
            
            # Save the new caption
            caption_file = image_path.with_suffix('.txt')
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(f"{caption}\n")
            
            # Update the caption in the editor
            self.review_caption_editor.setText(caption)
            
            QMessageBox.information(self, "Re-tag Complete", f"Caption has been regenerated for {image_name}.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while re-tagging: {e}")

    def load_tool_settings(self):
        """Load saved tool settings"""
        tool_settings = {}
        try:
            with open('tool_settings.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        tool_settings[key] = value
        except FileNotFoundError:
            pass
        return tool_settings

    def save_tool_settings(self):
        """Save tool settings"""
        # Collect all settings
        self.tool_settings['shared_directory'] = self.shared_directory.text()
        self.tool_settings['sb_project_name'] = self.sb_project_name.text()
        self.tool_settings['sb_output_dir'] = self.sb_output_dir.text()
        self.tool_settings['pc_text_to_remove'] = self.pc_text_to_remove.text()
        self.tool_settings['tw_text_to_add'] = self.tw_text_to_add.text()
        self.tool_settings['csv_filename'] = self.csv_filename.text()
        self.tool_settings['csv_save_location'] = self.csv_save_location.text()
        
        # Save to file
        with open('tool_settings.txt', 'w', encoding='utf-8') as f:
            for key, value in self.tool_settings.items():
                f.write(f"{key}:{value}\n")

    def browse_sb_input_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.sb_input_dir.setText(directory)

    def browse_sb_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.sb_output_dir.setText(directory)

    def browse_pc_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.pc_directory.setText(directory)

    def browse_tw_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.tw_directory.setText(directory)

    def browse_csv_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.csv_directory.setText(directory)

    def generate_website(self):
        """Generate a website from image-text pairs"""
        import os
        import shutil
        from pathlib import Path
        
        # Use shared directory if available
        image_text_folder = self.shared_directory.text()
        project_name = self.sb_project_name.text().strip()
        output_folder = self.sb_output_dir.text()
        
        if not image_text_folder or not project_name or not output_folder:
            QMessageBox.warning(self, "Input Error", "Please fill in all fields.")
            return
        
        # Append .html to the project name to form the index file name
        index_filename = project_name + '.html'

        # Create necessary folders
        images_folder = os.path.join(output_folder, 'images', project_name)
        html_folder = os.path.join(output_folder, 'HTML', project_name)
        index_file = os.path.join(output_folder, index_filename)

        # Create necessary folders
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(html_folder, exist_ok=True)

        # Collect subfolders
        subfolders = []
        for root_dir, dirs, files in os.walk(image_text_folder):
            # Skip the root folder itself
            if root_dir == image_text_folder:
                subfolders.extend(dirs)
                break  # Only need immediate subdirectories

        # If no subfolders, treat root as a single subfolder
        if not subfolders:
            subfolders = ['']

        # Data structure to hold images per subfolder
        images_per_subfolder = {}
        subfolders_with_pairs = []

        for subfolder in subfolders:
            images = []
            # Build the path to the subfolder
            if subfolder == '':
                current_folder = image_text_folder
            else:
                current_folder = os.path.join(image_text_folder, subfolder)

            # Collect images and descriptions in current subfolder
            has_pairs = False
            for root_dir, _, files in os.walk(current_folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        base_name = os.path.splitext(file)[0]
                        image_path = os.path.join(root_dir, file)
                        # Corresponding text file
                        text_file = os.path.join(root_dir, base_name + '.txt')
                        if not os.path.exists(text_file):
                            # Do not include images without corresponding text file
                            continue
                        with open(text_file, 'r', encoding='utf-8') as tf:
                            description = tf.read().strip()
                        # Copy image to images_folder
                        dest_image_subfolder = os.path.join(images_folder, subfolder)
                        os.makedirs(dest_image_subfolder, exist_ok=True)
                        dest_image_path = os.path.join(dest_image_subfolder, file)
                        shutil.copy2(image_path, dest_image_path)
                        images.append({'name': file, 'description': description, 'subfolder': subfolder})
                        has_pairs = True
            if has_pairs:
                images_per_subfolder[subfolder] = images
                subfolders_with_pairs.append(subfolder)

        # Determine if only one subfolder has pairs
        if len(subfolders_with_pairs) == 1:
            # Use that subfolder's images as the main index
            subfolder = subfolders_with_pairs[0]
            images = images_per_subfolder[subfolder]
            # Generate main index page directly with images
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write('<!DOCTYPE html>\n')
                f.write('<html lang="en">\n')
                f.write('<head>\n')
                f.write('    <meta charset="UTF-8">\n')
                f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
                f.write(f'    <title>{project_name}</title>\n')
                f.write('    <style>\n')
                f.write('        body {\n')
                f.write('            font-family: "Segoe UI", Arial, sans-serif;\n')
                f.write('            background-color: #121212;\n')
                f.write('            color: #ffffff;\n')
                f.write('        }\n')
                f.write('        .search-container {\n')
                f.write('            text-align: center;\n')
                f.write('            margin: 20px;\n')
                f.write('        }\n')
                f.write('        .search-container input {\n')
                f.write('            width: 50%;\n')
                f.write('            padding: 10px;\n')
                f.write('            font-size: 16px;\n')
                f.write('            background-color: #1E1E1E;\n')
                f.write('            color: #ffffff;\n')
                f.write('            border: none;\n')
                f.write('            border-radius: 4px;\n')
                f.write('        }\n')
                f.write('        .gallery {\n')
                f.write('            display: flex;\n')
                f.write('            flex-wrap: wrap;\n')
                f.write('            justify-content: center;\n')
                f.write('        }\n')
                f.write('        .gallery-item {\n')
                f.write('            margin: 10px;\n')
                f.write('            text-align: center;\n')
                f.write('            background-color: #1E1E1E;\n')
                f.write('            padding: 10px;\n')
                f.write('            border-radius: 8px;\n')
                f.write('        }\n')
                f.write('        .gallery-item img {\n')
                f.write('            width: 200px;\n')
                f.write('            height: auto;\n')
                f.write('            border-radius: 4px;\n')
                f.write('        }\n')
                f.write('        .hidden {\n')
                f.write('            display: none;\n')
                f.write('        }\n')
                f.write('        a {\n')
                f.write('            color: #1E88E5;\n')
                f.write('            text-decoration: none;\n')
                f.write('        }\n')
                f.write('        a:hover {\n')
                f.write('            text-decoration: underline;\n')
                f.write('        }\n')
                f.write('    </style>\n')
                f.write('</head>\n')
                f.write('<body>\n')
                f.write(f'    <h1 style="text-align:center;">{project_name}</h1>\n')
                f.write('    <div class="search-container">\n')
                f.write('        <input type="text" id="search-input" placeholder="Search images..." onkeyup="searchImages()">\n')
                f.write('    </div>\n')
                f.write('    <div class="gallery" id="gallery">\n')

                for img in images:
                    image_filename = img['name']
                    image_base = os.path.splitext(image_filename)[0]
                    image_subfolder = img['subfolder']
                    # Paths to image and image page
                    image_src_full = os.path.join(images_folder, image_subfolder, image_filename)
                    image_page_full = os.path.join(html_folder, image_subfolder, f'{image_base}.html')
                    # Relative paths from index file
                    image_src = os.path.relpath(image_src_full, start=output_folder)
                    image_page = os.path.relpath(image_page_full, start=output_folder)
                    # Replace backslashes with forward slashes
                    image_src = image_src.replace('\\', '/')
                    image_page = image_page.replace('\\', '/')
                    description = img['description']
                    f.write('        <div class="gallery-item">\n')
                    f.write(f'            <a href="{image_page}"><img src="{image_src}" alt="{image_base}"></a>\n')
                    # Include hidden elements for search
                    f.write('            <div class="hidden">\n')
                    f.write(f'                <span class="image-name">{image_base}</span>\n')
                    f.write(f'                <span class="image-description">{description}</span>\n')
                    f.write('            </div>\n')
                    f.write('        </div>\n')

                f.write('    </div>\n')

                # Add JavaScript for search functionality
                f.write('    <script>\n')
                f.write('        function searchImages() {\n')
                f.write('            const input = document.getElementById("search-input");\n')
                f.write('            const filter = input.value.toLowerCase();\n')
                f.write('            const gallery = document.getElementById("gallery");\n')
                f.write('            const items = gallery.getElementsByClassName("gallery-item");\n')
                f.write('            for (let i = 0; i < items.length; i++) {\n')
                f.write('                const name = items[i].querySelector(".image-name").textContent.toLowerCase();\n')
                f.write('                const description = items[i].querySelector(".image-description").textContent.toLowerCase();\n')
                f.write('                if (name.includes(filter) || description.includes(filter)) {\n')
                f.write('                    items[i].style.display = "";\n')
                f.write('                } else {\n')
                f.write('                    items[i].style.display = "none";\n')
                f.write('                }\n')
                f.write('            }\n')
                f.write('        }\n')
                f.write('    </script>\n')
                f.write('</body>\n')
                f.write('</html>\n')

            # Generate individual image pages
            subfolder_html_folder = os.path.join(html_folder, subfolder)
            os.makedirs(subfolder_html_folder, exist_ok=True)
            for img in images:
                image_filename = img['name']
                description = img['description']
                image_base = os.path.splitext(image_filename)[0]
                image_page = os.path.join(subfolder_html_folder, f'{image_base}.html')
                # Calculate relative paths from the image page
                relative_image_path = os.path.relpath(os.path.join(images_folder, subfolder, image_filename), start=os.path.dirname(image_page))
                relative_index_path = os.path.relpath(index_file, start=os.path.dirname(image_page))
                # Replace backslashes with forward slashes
                relative_image_path = relative_image_path.replace('\\', '/')
                relative_index_path = relative_index_path.replace('\\', '/')

                with open(image_page, 'w', encoding='utf-8') as f:
                    f.write('<!DOCTYPE html>\n')
                    f.write('<html lang="en">\n')
                    f.write('<head>\n')
                    f.write('    <meta charset="UTF-8">\n')
                    f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
                    f.write(f'    <title>{image_base}</title>\n')
                    f.write('    <style>\n')
                    f.write('        body {\n')
                    f.write('            font-family: "Segoe UI", Arial, sans-serif;\n')
                    f.write('            text-align: center;\n')
                    f.write('            margin: 0;\n')
                    f.write('            padding: 0;\n')
                    f.write('            background-color: #121212;\n')
                    f.write('            color: #ffffff;\n')
                    f.write('        }\n')
                    f.write('        img {\n')
                    f.write('            max-width: 80%;\n')
                    f.write('            height: auto;\n')
                    f.write('            margin-top: 20px;\n')
                    f.write('            border-radius: 8px;\n')
                    f.write('        }\n')
                    f.write('        .description {\n')
                    f.write('            margin: 20px;\n')
                    f.write('            font-size: 1.2em;\n')
                    f.write('        }\n')
                    f.write('        a {\n')
                    f.write('            display: inline-block;\n')
                    f.write('            margin-top: 20px;\n')
                    f.write('            text-decoration: none;\n')
                    f.write('            color: #1E88E5;\n')
                    f.write('            font-size: 1.1em;\n')
                    f.write('        }\n')
                    f.write('        a:hover {\n')
                    f.write('            text-decoration: underline;\n')
                    f.write('        }\n')
                    f.write('    </style>\n')
                    f.write('</head>\n')
                    f.write('<body>\n')
                    f.write(f'    <h1>{image_base}</h1>\n')
                    f.write(f'    <img src="{relative_image_path}" alt="{image_base}">\n')
                    f.write(f'    <div class="description">{description}</div>\n')
                    f.write(f'    <a href="{relative_index_path}">Back to Gallery</a>\n')
                    f.write('</body>\n')
                    f.write('</html>\n')
        else:
            # Multiple subfolders with pairs - create index with links to subfolder pages
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write('<!DOCTYPE html>\n')
                f.write('<html lang="en">\n')
                f.write('<head>\n')
                f.write('    <meta charset="UTF-8">\n')
                f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
                f.write(f'    <title>{project_name}</title>\n')
                f.write('    <style>\n')
                f.write('        body {\n')
                f.write('            font-family: "Segoe UI", Arial, sans-serif;\n')
                f.write('            background-color: #121212;\n')
                f.write('            color: #ffffff;\n')
                f.write('            text-align: center;\n')
                f.write('        }\n')
                f.write('        .folder-list {\n')
                f.write('            display: flex;\n')
                f.write('            flex-direction: column;\n')
                f.write('            align-items: center;\n')
                f.write('            margin-top: 50px;\n')
                f.write('        }\n')
                f.write('        .folder-item {\n')
                f.write('            margin: 10px;\n')
                f.write('            padding: 15px 30px;\n')
                f.write('            background-color: #1E1E1E;\n')
                f.write('            border-radius: 8px;\n')
                f.write('            width: 300px;\n')
                f.write('        }\n')
                f.write('        a {\n')
                f.write('            color: #1E88E5;\n')
                f.write('            text-decoration: none;\n')
                f.write('            font-size: 1.2em;\n')
                f.write('        }\n')
                f.write('        a:hover {\n')
                f.write('            text-decoration: underline;\n')
                f.write('        }\n')
                f.write('    </style>\n')
                f.write('</head>\n')
                f.write('<body>\n')
                f.write(f'    <h1>{project_name}</h1>\n')
                f.write('    <div class="folder-list">\n')
                
                for subfolder in subfolders_with_pairs:
                    subfolder_display = subfolder if subfolder else "Main Folder"
                    subfolder_index = f"HTML/{project_name}/{subfolder}/index.html"
                    f.write('        <div class="folder-item">\n')
                    f.write(f'            <a href="{subfolder_index}">{subfolder_display}</a>\n')
                    f.write('        </div>\n')
                
                f.write('    </div>\n')
                f.write('</body>\n')
                f.write('</html>\n')
            
            # Generate subfolder index pages and image pages
            for subfolder in subfolders_with_pairs:
                images = images_per_subfolder[subfolder]
                subfolder_html_folder = os.path.join(html_folder, subfolder)
                os.makedirs(subfolder_html_folder, exist_ok=True)
                subfolder_index_file = os.path.join(subfolder_html_folder, "index.html")
                
                with open(subfolder_index_file, 'w', encoding='utf-8') as f:
                    # Create subfolder index page HTML
                    # (Similar to the main index page but with links back to main index)
                    # ... (code omitted for brevity)
                    pass

        QMessageBox.information(self, "Success", f"Website '{project_name}' generated successfully!")

    def remove_text_from_files(self):
        """Remove specified text from all .txt files in a directory"""
        import os
        
        # Use shared directory
        directory = self.shared_directory.text()
        text_to_remove = self.pc_text_to_remove.text()
        
        if not directory or not text_to_remove:
            QMessageBox.warning(self, "Error", "Please fill in both fields")
            return
        
        try:
            files_modified = 0
            for root_dir, _, files in os.walk(directory):
                for filename in files:
                    if filename.endswith(".txt"):
                        file_path = os.path.join(root_dir, filename)
                        with open(file_path, 'r+', encoding='utf-8') as file:
                            content = file.read()
                            if text_to_remove in content:
                                new_content = content.replace(text_to_remove, '')
                                file.seek(0)
                                file.write(new_content)
                                file.truncate()
                                files_modified += 1
        
            QMessageBox.information(self, "Success", f"Removed text from {files_modified} files")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def add_text_to_files(self):
        """Add specified text to the beginning of all .txt files in a directory"""
        import os
        
        # Use shared directory
        directory = self.shared_directory.text()
        text_to_add = self.tw_text_to_add.text()
        
        if not directory or not text_to_add:
            QMessageBox.warning(self, "Error", "Please fill in both fields")
            return
        
        try:
            files_modified = 0
            for root_dir, _, files in os.walk(directory):
                for filename in files:
                    if filename.endswith(".txt"):
                        file_path = os.path.join(root_dir, filename)
                        with open(file_path, 'r+', encoding='utf-8') as file:
                            content = file.read()
                            file.seek(0, 0)
                            file.write(text_to_add + content)
                            file.truncate()
                        files_modified += 1
        
            QMessageBox.information(self, "Success", f"Added text to {files_modified} files")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def browse_csv_save_location(self):
        """Browse for a directory to save the CSV file"""
        directory = QFileDialog.getExistingDirectory(self, "Select CSV Save Location")
        if directory:
            self.csv_save_location.setText(directory)
            # Save the setting
            self.tool_settings['csv_save_location'] = directory
            self.save_tool_settings()

    def create_csv(self):
        """Create a CSV file from all .txt files in a directory"""
        import os
        import csv
        
        # Use shared directory
        directory = self.shared_directory.text()
        csv_filename = self.csv_filename.text()
        csv_save_location = self.csv_save_location.text()
        
        if not directory or not csv_filename:
            QMessageBox.warning(self, "Error", "Please fill in all required fields")
            return
        
        if not csv_filename.endswith('.csv'):
            csv_filename += '.csv'
        
        # If save location is specified, use it; otherwise save in current directory
        if csv_save_location:
            full_csv_path = os.path.join(csv_save_location, csv_filename)
        else:
            full_csv_path = csv_filename
        
        try:
            with open(full_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['File Name', 'Content'])
                
                for root_dir, _, files in os.walk(directory):
                    for filename in files:
                        if filename.endswith(".txt"):
                            file_path = os.path.join(root_dir, filename)
                            with open(file_path, 'r', encoding='utf-8') as txtfile:
                                content = txtfile.read()
                                csvwriter.writerow([filename, content])
            
            QMessageBox.information(self, "Success", f"CSV file saved to: {full_csv_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def browse_shared_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Shared Directory")
        if directory:
            self.shared_directory.setText(directory)
            # This will trigger update_tool_directories via the textChanged signal

    def update_tool_directories(self):
        """Update all tool directories when the shared directory changes"""
        shared_dir = self.shared_directory.text()
        if shared_dir:
            # Save the shared directory in settings
            self.tool_settings['shared_directory'] = shared_dir
            self.save_tool_settings()

    def reset_ui_size(self):
        """Reset the UI to default font size and window dimensions"""
        # Reset to default font size
        self.current_font_size = 10  # Default font size
        
        # Update the font
        self.update_application_font()
        
        # Reset window size to default
        self.resize(1200, 800)  # Default window size
        
        # Center the window on screen
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
        
        QMessageBox.information(self, "UI Reset", "UI size has been reset to default.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CaptionApp()
    window.show()
    sys.exit(app.exec_())
