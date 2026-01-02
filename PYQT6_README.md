# ROCA Orbital Memory System - PyQt6 Edition

A modern PyQt6 implementation of the ROCA (Relevance, Orbital, Capsule, Analysis) knowledge visualization system. This version replaces the Pygame-based visualization with a native PyQt6 GUI featuring real-time orbital mechanics, interactive capsule management, and advanced memory operations.

## Features

### Core Features
- **Orbital Visualization**: Real-time animated visualization of knowledge capsules orbiting a central core
- **Knowledge Capsules**: Manage discrete units of knowledge with metadata (character, kind, certainty, etc.)
- **Memory Dynamics**: Automatic orbit updates, gravitational influences, and capsule merging
- **Interactive UI**: Add, view, and manage capsules through an intuitive interface
- **GPU Acceleration Ready**: NVIDIA and Apple Silicon GPU support (when configured)

### PyQt6 Advantages Over Pygame
- Native operating system look and feel
- Better support for complex UI layouts
- Multi-threaded operations without blocking the UI
- Easier scaling and responsive design
- Better font rendering and text handling
- Native file dialogs and system integration

### UI Components
1. **Orbital Canvas**: Central visualization showing capsules in orbital patterns
2. **Statistics Panel**: Real-time memory statistics and system status
3. **Control Panel**: Buttons for memory operations (add, update, generate hypotheses)
4. **Capsule Table**: Detailed view of all capsules with sortable columns
5. **Tabbed Interface**: Easy switching between different views

## Installation

### Requirements
- Python 3.8+
- PyQt6
- NumPy
- PyTorch (for GPU support and similarity calculations)

### Setup

```bash
# Install PyQt6 and dependencies
pip install PyQt6 numpy torch

# Optional: For GPU acceleration
pip install cupy-cuda11x  # Replace 11x with your CUDA version
pip install speech-recognition  # For voice input support
```

## Usage

### Basic Usage

```python
from roca_orbital_pyqt6 import RocaOrbitalMemoryApp
import sys
from PyQt6.QtWidgets import QApplication

# Create and run the application
app = QApplication(sys.argv)
window = RocaOrbitalMemoryApp()
window.show()
sys.exit(app.exec())
```

### Running from Command Line

```bash
python roca_orbital_pyqt6.py
```

## Interface Guide

### Main Window
The application window is divided into two main sections:

1. **Left Panel (Orbital Visualization)**
   - Shows capsules orbiting the ROCA core
   - Capsules animate continuously
   - Click on a capsule to select it
   - Different colors indicate different capsule properties:
     - Cyan: Cayde-generated capsules
     - Yellow: High insight potential
     - Purple: Theories proven later
     - Red-ish: Regular capsules with varying certainty

2. **Right Panel (Controls & Information)**
   - **Statistics Tab**: View memory statistics
   - **Controls Tab**: Add capsules, update memory, generate hypotheses
   - **Capsules Tab**: Table view of all capsules

### Adding a Capsule

1. Click "Add Capsule" button
2. Fill in the dialog:
   - **Content**: The knowledge to add
   - **Character**: Attribution (e.g., "Einstein", "Newton")
   - **Kind**: Type of knowledge (Theory, Method, Observation, Concept, Hypothesis)
   - **Certainty**: Confidence level (0-1)
3. Click "Add" to add to memory

### Memory Operations

- **Update Memory**: Triggers orbit updates and applies gravitational influences
- **Generate Hypotheses**: Creates new hypothesis capsules
- **Auto-Update**: Checkbox to automatically update memory every second
- **Clear Memory**: Remove all capsules

## Key Classes

### Capsule
Represents a unit of knowledge with metadata.

```python
capsule = Capsule(
    content="Einstein's Theory of Relativity",
    character="Einstein",
    kind=CapsuleKind.THEORY,
    certainty=0.9,
    perspective="scientific"
)
```

### RocaMemory
Core memory management system.

```python
memory = RocaMemory()
capsule = memory.add_capsule(
    content="Knowledge content",
    character="source",
    kind=CapsuleKind.CONCEPT
)
memory.orbit_update()
```

### RocaOrbitalMemoryApp
Main PyQt6 application window with all UI components.

## Configuration

### Color Scheme
The application uses a dark theme with the following color mapping:
- Background: Dark blue-gray (#141e1e)
- Core: Bright cyan (100, 200, 255)
- Orbital rings: Subtle purple (60, 60, 80)
- Text: White

### Animation Settings
- Orbital period varies by orbit level
- Animation runs at ~60 FPS
- Smooth interpolation between states

### GPU Detection
The application automatically detects and reports:
- **NVIDIA GPU**: CUDA support
- **Apple Silicon**: Metal Performance Shaders
- **CPU**: Fallback mode

## Advanced Features

### Batch Operations
The RocaMemory class supports batch operations for efficiency:
- Merge similar capsules
- Split conflicting capsules
- Generate multiple hypotheses
- Apply gravitational influences

### Extensibility
The system is designed for easy extension:

1. **Custom Capsule Types**: Extend the `CapsuleKind` enum
2. **Custom Visualizations**: Subclass `OrbitalCanvasWidget`
3. **Custom Dialogs**: Extend `AddCapsuleDialog`
4. **Memory Algorithms**: Extend `RocaMemory` class

## Performance Considerations

### Rendering
- Uses Qt's native rendering engine
- Smooth animation at 60 FPS
- Efficient circle drawing with antialiasing

### Memory Management
- Capsules stored in a list for O(1) access
- Minimal memory overhead per capsule
- Automatic garbage collection of old capsules

### Scaling
- Tested with 100+ capsules
- Linear time complexity for orbit updates
- Efficient collision detection for selection

## Troubleshooting

### PyQt6 Installation Issues
If you encounter import errors:
```bash
pip install --upgrade PyQt6
```

### GPU Not Detected
The system gracefully falls back to CPU. Check:
```python
from roca_orbital_pyqt6 import GPU_AVAILABLE, GPU_TYPE
print(f"GPU Available: {GPU_AVAILABLE}, Type: {GPU_TYPE}")
```

### Slow Animation
Reduce update frequency or increase timer interval in `OrbitalCanvasWidget`:
```python
self.timer.start(32)  # ~30 FPS instead of 60
```

## File Structure

```
Roca_Code/
├── Roca_orbital.py              # Original Pygame version
└── Roca_orbital_pyqt6.py        # New PyQt6 version
```

## Comparison: PyQt6 vs Pygame

| Feature | PyQt6 | Pygame |
|---------|-------|--------|
| Native Look & Feel | ✓ | ✗ |
| UI Complexity | Excellent | Limited |
| Performance | Excellent | Good |
| Learning Curve | Moderate | Easy |
| Cross-platform | ✓ | ✓ |
| Accessibility | ✓ | Limited |

## Future Enhancements

- [ ] Real-time voice input for capsule creation
- [ ] 3D orbital visualization with WebGL
- [ ] Historical playback of memory evolution
- [ ] Network graph view of capsule relationships
- [ ] Export/import functionality
- [ ] Customizable color themes
- [ ] Performance profiling tools
- [ ] Integration with original Cayde personality system

## License

Same as the original ROCA Orbital Memory System

## Contributing

Contributions are welcome! Areas for improvement:
- Enhanced visualization features
- Additional statistical analysis tools
- Integration with the original system's mathematical engines
- Performance optimizations
- Additional input methods (voice, natural language)

## Contact

For issues, feature requests, or questions about the PyQt6 version, please refer to the main project repository.

---

**Note**: This PyQt6 version maintains API compatibility with the original system while providing a modern, responsive user interface suitable for advanced knowledge management and orbital visualization.
