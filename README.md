

```markdown
# Joy Caption Alpha Two - GUI Edition

A comprehensive fork of [Joy Caption Alpha Two](https://github.com/D3voz/joy-caption-alpha-two-gui-mod) that adds an enhanced dark mode GUI, extensive batch processing capabilities, and numerous quality-of-life improvements. This modification streamlines the image captioning workflow while maintaining the powerful core functionality of the original project.

## 🌟 New Features

### Enhanced Interface
- **Dark Mode by Default**: Professional dark theme interface
- **Adjustable UI**:
  - Dynamic font size controls
  - Window scaling
  - Persistent size settings
- **Three-Tab Layout**:
  - Tagging: Main captioning interface
  - Review: Caption management
  - Tools: Batch processing utilities

### Advanced Captioning
- **Multiple Caption Types**:
  - Descriptive (Formal/Informal)
  - Training Prompt
  - MidJourney Style
  - Booru Tags
  - Art Critic Analysis
  - Product Listing
  - Social Media Ready
- **Caption Customization**:
  - Adjustable length (Short/Medium/Long)
  - Extra context options
  - Character/person name integration
  - Custom prompt system
  - Saved prompts management

### Batch Processing
- **Directory Operations**:
  - Recursive directory scanning
  - Bulk caption generation
  - Mass text manipulation
- **Text Tools**:
  - Add/remove text from multiple files
  - Trigger word integration
  - Batch text prepending/appending
- **Export Options**:
  - CSV export functionality
  - Structured data output
  - Custom save locations

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- CUDA-compatible GPU (optional)
- Windows 10/11 (Primary Support)

### Quick Install (Windows)
1. Clone the repository:
```bash
git clone https://github.com/yourusername/joy-caption-alpha-two-gui-mod.git
cd joy-caption-alpha-two-gui-mod
```

2. Run the automatic setup:
```bash
Joy_run.bat
```

### Manual Installation
1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the environment:
```bash
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch the application:
```bash
python dark_mode_gui.py
```

## 🎮 Usage Guide

### Tagging Tab
1. **Image Selection**:
   - Single image selection
   - Directory batch processing
   - Drag and drop support

2. **Caption Configuration**:
   - Select caption type
   - Adjust length
   - Enable extra options
   - Add character names
   - Use custom prompts

3. **Processing**:
   - Generate single captions
   - Batch process directories
   - Monitor progress

### Review Tab
- Browse processed images
- Edit generated captions
- Save modifications
- Re-tag capabilities
- Bulk editing tools

### Tools Tab
- **Text Operations**:
  - Add/remove text
  - Manage trigger words
  - Batch rename
  - CSV conversion
- **Directory Management**:
  - Set shared directories
  - Configure outputs
  - Manage organization

## ⚙️ Configuration

The application maintains two settings files:
- `app_settings.txt`: Interface and general preferences
- `tool_settings.txt`: Tool-specific configurations

### Model Configuration
- Automatic model download
- Local caching
- CUDA acceleration support

## 🔧 Troubleshooting

Common solutions:

1. **Startup Issues**:
   - Verify Python installation
   - Check virtual environment
   - Confirm dependencies

2. **Performance**:
   - Enable CUDA
   - Reduce batch size
   - Close background apps

3. **Memory Issues**:
   - Reduce batch size
   - Close other applications
   - Check system requirements

## 🙏 Acknowledgments

This project is a fork of the original [Joy Caption Alpha Two](https://github.com/D3voz/joy-caption-alpha-two-gui-mod) by D3voz. Special thanks to:
- Original Joy Caption Alpha Two team
- PyQt5 development community
- Hugging Face for model hosting
- All contributors and testers

## 📄 License

This project maintains the original license from Joy Caption Alpha Two while adding these modifications under the same terms.

## 📞 Support

For issues and feature requests, please use the GitHub issues tracker.
```

Would you like me to expand on any particular section or add more specific technical details?
