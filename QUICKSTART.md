# ROCA Orbital Memory System - PyQt6 Quick Start Guide

## Overview

You now have **two PyQt6 implementations** of the ROCA Orbital Memory System:

1. **Roca_orbital_pyqt6.py** - Clean, minimal implementation focused on core visualization
2. **Roca_orbital_enhanced.py** - Full-featured version with document ingestion, querying, and more

## Installation

```bash
# Install PyQt6 and dependencies
pip install PyQt6 numpy torch

# Optional: For GPU acceleration  
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

## Running the Applications

### Basic Version (Recommended for learning)

```bash
python roca_orbital_pyqt6.py
```

**Features:**
- Real-time orbital visualization
- Add capsules via dialog
- Memory statistics dashboard
- Table view of all capsules
- Auto-update capability

### Enhanced Version (Full functionality)

```bash
python roca_orbital_enhanced.py
```

**Features:**
- Everything in the basic version, plus:
- Document ingestion (extract knowledge from text)
- Memory querying (ask questions about stored knowledge)
- Hypothesis generation
- Advanced orbital dynamics with gravitational influence
- Capsule merging for similar knowledge
- Menu bar with file/edit/view/help
- Personality trait system (from original Cayde)

## Key Differences from Pygame Version

| Feature | Pygame | PyQt6 |
|---------|--------|-------|
| Platform Integration | Generic | Native OS look & feel |
| UI Responsiveness | Good | Excellent |
| Text Rendering | Basic | Professional |
| Extensibility | Limited | Excellent |
| System Integration | Minimal | Full (file dialogs, etc.) |
| Multi-threading | Manual | Built-in |

## User Interface Guide

### Main Window Layout

```
┌─────────────────────────────────────────────────────────┐
│ Menu Bar (File, Edit, View, Help)                       │
├─────────────────────────┬───────────────────────────────┤
│                         │ ┌─────────────────────────────┤
│   ORBITAL CANVAS        │ │ STATISTICS / CONTROLS / TABLE
│   (70% width)           │ │ (30% width)
│   - Real-time animation │ │ - Tabs for different views
│   - Capsule orbits      │ │ - Memory stats
│   - Interactive clicks  │ │ - Control buttons
│                         │ │ - Capsule table
└─────────────────────────┴───────────────────────────────┘
```

### Orbital Canvas Features

**Colors:**
- **Cyan**: Cayde-generated capsules (AI hypotheses)
- **Yellow**: High insight potential (>0.7)
- **Purple**: Theories proven later
- **Gradient**: Regular capsules (color based on certainty and gravity)

**Animation:**
- Capsules orbit around central core
- Orbit speed varies by orbital level
- Smooth 60 FPS animation

**Interaction:**
- Click on orbital rings to see statistics
- Visual feedback for selected capsules

### Controls Tab

**Memory Operations:**
- **Add Capsule**: Create new knowledge units
- **Update Memory**: Trigger orbital dynamics and influence calculations
- **Generate Hypotheses**: Create AI-generated theories based on trends
- **Ingest Document**: Extract knowledge from text files
- **Query Memory**: Ask questions about stored knowledge
- **Clear Memory**: Reset the entire system

**Auto-Update:**
- Toggle to automatically update memory every second
- Useful for watching dynamic evolution of knowledge

### Statistics Tab

**Displays:**
- Total number of capsules
- Core capsules (high gravity)
- Average certainty level
- Average gravity level
- Number of high-insight capsules
- GPU availability status

### Capsules Tab

**Features:**
- Table view of all capsules
- Sortable columns: Content, Character, Kind, Certainty, Gravity, Status
- Refresh button to update view
- Easily browse entire memory

## Workflow Examples

### Example 1: Building a Knowledge Base

1. Click "Add Capsule"
2. Enter: "F = ma" (Newton's Second Law)
   - Character: Newton
   - Kind: Theory
   - Certainty: 0.9
3. Repeat for more concepts
4. Click "Update Memory" to see orbital dynamics
5. Watch certainty values evolve in the table

### Example 2: Ingesting a Document

1. Click "Ingest Document"
2. Paste text (e.g., a Wikipedia article)
3. Specify author (or auto-detect)
4. System automatically creates capsules from sentences
5. View generated capsules in the table

### Example 3: Querying Your Knowledge

1. Click "Query Memory"
2. Ask a question: "What is relativity?"
3. System finds relevant capsules
4. Returns answer based on stored knowledge

### Example 4: Generating Hypotheses

1. Add several related capsules first
2. Click "Generate Hypotheses"
3. System combines insights to create new hypotheses
4. Hypotheses appear in cyan color in the orbit
5. Mark promising ones for further investigation

## Advanced Usage

### Customizing the Visualization

Edit colors in the canvas painting function:

```python
# In OrbitalCanvasWidget.paintEvent()
if capsule.character == "cayde":
    color = QColor(0, 255, 255)  # Cyan - change RGB values
```

### Adjusting Animation Speed

```python
# In OrbitalCanvasWidget.__init__()
self.timer.start(32)  # Lower = faster (milliseconds between frames)
```

### Modifying Orbital Dynamics

```python
# In EnhancedRocaMemory.orbit_update()
capsule.orbit_radius += random.uniform(-0.02, 0.02)  # Change amount
```

### Creating Custom Capsule Types

```python
# In CapsuleKind enum
class CapsuleKind(Enum):
    THEORY = "theory"
    # Add your custom type:
    INSIGHT = "insight"
```

## Performance Tips

1. **Large Memory**: With 1000+ capsules, consider:
   - Filtering by character or kind
   - Archiving old capsules
   - Using the Core Capsules view

2. **GPU Usage**: PyQt6 automatically uses GPU for:
   - Capsule embedding calculations
   - Similarity computations
   - Parallel orbit updates

3. **Animation**: For slower machines:
   - Reduce frame rate: `self.timer.start(32)` → `self.timer.start(64)`
   - Disable auto-update
   - Use Static view instead of animated

## Troubleshooting

### PyQt6 Won't Install
```bash
pip install --upgrade PyQt6
# Or specify version:
pip install PyQt6==6.5.0
```

### GPU Not Detected
The system gracefully falls back to CPU. Check:
```python
from roca_orbital_enhanced import GPU_AVAILABLE, GPU_TYPE
print(f"GPU: {GPU_TYPE}, Available: {GPU_AVAILABLE}")
```

### Slow Visualization
- Check GPU_AVAILABLE status
- Reduce animation frame rate
- Limit number of visible capsules

### Memory Leaks
Clear old capsules regularly:
```python
# Keep only core and recent capsules
self.memory.capsules = self.memory.get_core_capsules()
```

## Migrating from Pygame Version

The PyQt6 versions maintain similar core concepts:
- Capsule structure is identical
- Orbital mechanics are equivalent
- Color scheme is similar
- APIs are compatible

**To migrate code:**
```python
# Old Pygame version
from roca_orbital import RocaOrbitalMemory
memory = RocaOrbitalMemory()

# New PyQt6 version
from roca_orbital_enhanced import EnhancedRocaMemory
memory = EnhancedRocaMemory()
# Rest of code works the same!
```

## Next Steps

1. **Explore the UI**: Click around, add capsules, watch them orbit
2. **Read the code**: Both implementations are well-commented
3. **Customize**: Modify colors, orbital parameters, dynamics
4. **Integrate**: Use the memory system in your own applications
5. **Extend**: Add new features (persistence, networking, ML integration)

## Files Overview

```
Roca_Code/
├── Roca_orbital.py                 # Original Pygame version (reference)
├── Roca_orbital_pyqt6.py           # Basic PyQt6 implementation (40KB)
├── Roca_orbital_enhanced.py        # Full PyQt6 implementation (35KB)
├── PYQT6_README.md                 # Comprehensive documentation
└── QUICKSTART.md                   # This file
```

## Support & Feedback

- Check the PYQT6_README.md for detailed documentation
- Review the code comments for specific features
- The enhanced version includes personality traits from the original Cayde system
- Both versions are designed for extensibility

---

**Happy Exploring!** 🚀

The ROCA Orbital Memory System is now ready for knowledge visualization and management in a modern, responsive interface.
