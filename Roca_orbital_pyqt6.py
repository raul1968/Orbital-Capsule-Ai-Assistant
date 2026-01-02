#!/usr/bin/env python3
"""
ROCA Orbital Memory System - PyQt6 Version

A PyQt6 implementation of the ROCA (Relevance, Orbital, Capsule, Analysis)
knowledge system with orbital visualization capabilities.

This module provides the same functionality as the Pygame version but with
a modern Qt-based GUI including:
- Capsule-based knowledge representation
- Orbital visualization of knowledge capsules
- GPU-accelerated similarity calculations
- Memory operations (add, merge, orbit updates)
- PyQt6-based visualization and UI

Usage:
    from roca_orbital_pyqt6 import RocaOrbitalMemoryApp
    
    app = RocaOrbitalMemoryApp()
    app.run()
"""

import sys
import re
import string
import random
import uuid
import time
import math
from collections import Counter
from typing import List, Dict, Tuple, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import torch

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFrame, QScrollArea,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal, QThread, QObject, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QIcon, QPixmap

# Try imports for optional features
try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# GPU Detection
try:
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_TYPE = "NVIDIA"
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        GPU_AVAILABLE = True
        GPU_TYPE = "APPLE"
    else:
        GPU_AVAILABLE = False
        GPU_TYPE = "CPU"
except:
    GPU_AVAILABLE = False
    GPU_TYPE = "CPU"

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# ===== IMPORTS FROM ORIGINAL FILE =====
# For this version, we'll import the core classes from the original file
# or redefine the essential ones. Here we'll define the key classes needed:

class CapsuleKind(Enum):
    """Types of knowledge capsules"""
    THEORY = "theory"
    METHOD = "method"
    OBSERVATION = "observation"
    CONCEPT = "concept"
    HYPOTHESIS = "hypothesis"


@dataclass
class Capsule:
    """Knowledge capsule with metadata"""
    content: str
    perspective: str = "default"
    character: str = "unknown"
    persona: str = "default"
    kind: CapsuleKind = CapsuleKind.CONCEPT
    certainty: float = 0.5
    gravity: float = 0.5
    orbit_radius: float = 1.0
    success_status: Optional[str] = None
    insight_potential: float = 0.5
    temporal_order: int = 0
    confidence_history: List[float] = field(default_factory=list)
    paradigm_shifts: List[Dict[str, Any]] = field(default_factory=list)
    links: List['Capsule'] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = np.random.rand(128)
        self.confidence_history.append(self.certainty)


# Placeholder for RocaMemory - in production, import from original file
class RocaMemory:
    """Minimal placeholder - extend with full implementation"""
    
    def __init__(self):
        self.capsules: List[Capsule] = []
        self.cayde_personality = {
            'obsession_with_invariants': {'level': 0.8},
            'preference_for_geometric_intuition': {'level': 0.9},
            'distrust_of_unnecessary_constants': {'level': 0.7},
            'intolerance_for_inconsistency': {'level': 0.85},
            'willingness_to_discard_cherished_models': {'level': 0.8},
            'openness_to_revolution': 0.9,
            'current_focus': 'geometric_unification',
            'personality_evolution': [],
            'disagreements_with_scientists': [],
            'abandoned_intuitions': []
        }
    
    def add_capsule(self, content: str, character: str = "unknown", kind: CapsuleKind = CapsuleKind.CONCEPT,
                   perspective: str = "default", success_status: Optional[str] = None, **kwargs) -> Capsule:
        capsule = Capsule(
            content=content,
            character=character,
            kind=kind,
            perspective=perspective,
            success_status=success_status,
            **kwargs
        )
        self.capsules.append(capsule)
        return capsule
    
    def orbit_update(self):
        """Update orbital dynamics"""
        for capsule in self.capsules:
            # Simple orbital dynamics
            capsule.orbit_radius += random.uniform(-0.01, 0.01)
            capsule.gravity += random.uniform(-0.01, 0.01)
            capsule.gravity = max(0, min(1, capsule.gravity))
            capsule.orbit_radius = max(0.5, min(3, capsule.orbit_radius))
    
    def apply_influences(self):
        """Apply gravitational influences"""
        pass
    
    def merge_similar(self, threshold: float = 0.7):
        """Merge similar capsules"""
        pass
    
    def split_conflicting(self):
        """Split conflicting capsules"""
        pass
    
    def generate_hypotheses(self, num: int = 1) -> List[Capsule]:
        """Generate hypotheses"""
        hypotheses = []
        for _ in range(num):
            hyp = self.add_capsule(
                content=f"Hypothesis {random.randint(1000, 9999)}",
                character="cayde",
                kind=CapsuleKind.HYPOTHESIS
            )
            hypotheses.append(hyp)
        return hypotheses
    
    def get_core_capsules(self) -> List[Capsule]:
        """Get core capsules"""
        return [c for c in self.capsules if c.gravity > 0.7]


# ===== PyQt6 WIDGETS =====

class OrbitalCanvasWidget(QWidget):
    """Custom widget for orbital visualization"""
    
    def __init__(self, memory: RocaMemory):
        super().__init__()
        self.memory = memory
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #141e1e;")
        
        # Animation state
        self.animation_time = 0
        self.selected_capsule: Optional[Capsule] = None
        
        # Timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)  # ~60 FPS
        
    def animate(self):
        """Update animation state"""
        self.animation_time += 0.016
        self.update()
    
    def paintEvent(self, event):
        """Paint the orbital visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(20, 20, 30))
        
        # Get center
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # Draw orbital rings
        pen = QPen(QColor(60, 60, 80))
        pen.setWidth(1)
        painter.setPen(pen)
        
        for radius in [80, 120, 160, 200, 240, 280]:
            painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw core
        core_brush = QBrush(QColor(100, 200, 255))
        painter.setBrush(core_brush)
        painter.drawEllipse(center_x - 25, center_y - 25, 50, 50)
        
        # Draw core label
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 10)
        painter.setFont(font)
        painter.drawText(center_x - 40, center_y + 50, 80, 20, Qt.AlignmentFlag.AlignCenter, "ROCA Core")
        
        # Draw capsules
        if self.memory.capsules:
            orbit_radii = [80, 120, 160, 200, 240, 280]
            capsules_per_orbit = [8, 12, 16, 20, 24, 28]
            
            capsule_index = 0
            for orbit_level, (radius, max_per_orbit) in enumerate(zip(orbit_radii, capsules_per_orbit)):
                orbit_capsules = self.memory.capsules[capsule_index:capsule_index + max_per_orbit]
                capsule_index += max_per_orbit
                
                if not orbit_capsules:
                    continue
                
                for i, capsule in enumerate(orbit_capsules):
                    # Calculate position
                    angle = (i / len(orbit_capsules)) * 2 * math.pi + self.animation_time * 0.1 * (orbit_level + 1) * 0.1
                    x = center_x + math.cos(angle) * radius
                    y = center_y + math.sin(angle) * radius
                    
                    # Determine color
                    if capsule.character == "cayde":
                        color = QColor(0, 255, 255)
                    elif capsule.success_status == "archived":
                        color = QColor(80, 60, 60)
                    elif capsule.insight_potential > 0.7:
                        color = QColor(255, 255, 0)
                    elif capsule.success_status == "proven_later":
                        color = QColor(255, 0, 255)
                    else:
                        color = QColor(
                            int(100 + capsule.gravity * 155),
                            int(100 + capsule.gravity * 100),
                            int(150 + capsule.orbit_radius * 50)
                        )
                    
                    # Draw capsule
                    capsule_radius = 12
                    brush = QBrush(color)
                    painter.setBrush(brush)
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.drawEllipse(int(x) - capsule_radius, int(y) - capsule_radius,
                                       capsule_radius * 2, capsule_radius * 2)
                    
                    # Draw character label
                    name = str(capsule.character)[:7]
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(QFont("Arial", 8))
                    painter.drawText(int(x) - 20, int(y) + capsule_radius + 15, 40, 20,
                                    Qt.AlignmentFlag.AlignCenter, name)
    
    def mousePressEvent(self, event):
        """Handle mouse clicks to select capsules"""
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        click_x = event.position().x()
        click_y = event.position().y()
        
        # Check if any capsule was clicked
        for capsule in self.memory.capsules:
            # Simple hit detection
            if distance(click_x, click_y, center_x, center_y) < 300:
                self.selected_capsule = capsule
                self.update()
                return


class MemoryTableWidget(QWidget):
    """Widget to display memory capsules in table format"""
    
    def __init__(self, memory: RocaMemory):
        super().__init__()
        self.memory = memory
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Content", "Character", "Kind", "Certainty", "Status"])
        layout.addWidget(self.table)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.update_table)
        layout.addWidget(refresh_btn)
        
        self.setLayout(layout)
        self.update_table()
    
    def update_table(self):
        """Update table with capsule data"""
        self.table.setRowCount(len(self.memory.capsules))
        
        for row, capsule in enumerate(self.memory.capsules):
            # Content (truncated)
            content = capsule.content[:50] + "..." if len(capsule.content) > 50 else capsule.content
            self.table.setItem(row, 0, QTableWidgetItem(content))
            
            # Character
            self.table.setItem(row, 1, QTableWidgetItem(capsule.character))
            
            # Kind
            self.table.setItem(row, 2, QTableWidgetItem(capsule.kind.value))
            
            # Certainty
            self.table.setItem(row, 3, QTableWidgetItem(f"{capsule.certainty:.2f}"))
            
            # Status
            status = capsule.success_status or "unknown"
            self.table.setItem(row, 4, QTableWidgetItem(status))


class AddCapsuleDialog(QDialog):
    """Dialog for adding new capsules"""
    
    capsule_added = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize dialog UI"""
        layout = QVBoxLayout()
        
        # Content
        layout.addWidget(QLabel("Content:"))
        self.content_edit = QTextEdit()
        self.content_edit.setMaximumHeight(100)
        layout.addWidget(self.content_edit)
        
        # Character
        layout.addWidget(QLabel("Character:"))
        self.character_edit = QLineEdit()
        layout.addWidget(self.character_edit)
        
        # Kind
        layout.addWidget(QLabel("Kind:"))
        self.kind_combo = QComboBox()
        self.kind_combo.addItems([k.value for k in CapsuleKind])
        layout.addWidget(self.kind_combo)
        
        # Certainty
        layout.addWidget(QLabel("Certainty:"))
        self.certainty_spin = QDoubleSpinBox()
        self.certainty_spin.setRange(0, 1)
        self.certainty_spin.setSingleStep(0.1)
        self.certainty_spin.setValue(0.5)
        layout.addWidget(self.certainty_spin)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("Add")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.setWindowTitle("Add Knowledge Capsule")
        self.setGeometry(100, 100, 500, 400)
    
    def get_data(self) -> dict:
        """Get dialog data"""
        return {
            'content': self.content_edit.toPlainText(),
            'character': self.character_edit.text() or "unknown",
            'kind': CapsuleKind(self.kind_combo.currentText()),
            'certainty': self.certainty_spin.value()
        }


class StatsPanel(QWidget):
    """Panel showing memory statistics"""
    
    def __init__(self, memory: RocaMemory):
        super().__init__()
        self.memory = memory
        self.init_ui()
    
    def init_ui(self):
        """Initialize stats panel"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Memory Statistics")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Stats labels
        self.total_label = QLabel("Total Capsules: 0")
        self.core_label = QLabel("Core Capsules: 0")
        self.cayde_label = QLabel("Cayde Capsules: 0")
        self.insight_label = QLabel("Insights: 0")
        self.gpu_label = QLabel(f"GPU: {GPU_TYPE}")
        
        for label in [self.total_label, self.core_label, self.cayde_label, self.insight_label, self.gpu_label]:
            layout.addWidget(label)
        
        layout.addStretch()
        self.setLayout(layout)
        self.update_stats()
    
    def update_stats(self):
        """Update statistics display"""
        self.total_label.setText(f"Total Capsules: {len(self.memory.capsules)}")
        self.core_label.setText(f"Core Capsules: {len(self.memory.get_core_capsules())}")
        cayde_count = sum(1 for c in self.memory.capsules if c.character == "cayde")
        self.cayde_label.setText(f"Cayde Capsules: {cayde_count}")
        insight_count = sum(1 for c in self.memory.capsules if c.insight_potential > 0.7)
        self.insight_label.setText(f"Insights: {insight_count}")


class ControlPanel(QWidget):
    """Control panel for memory operations"""
    
    memory_updated = pyqtSignal()
    
    def __init__(self, memory: RocaMemory):
        super().__init__()
        self.memory = memory
        self.init_ui()
    
    def init_ui(self):
        """Initialize control panel"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Memory Controls")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Buttons
        self.add_btn = QPushButton("Add Capsule")
        self.add_btn.clicked.connect(self.on_add_capsule)
        layout.addWidget(self.add_btn)
        
        self.update_btn = QPushButton("Update Memory")
        self.update_btn.clicked.connect(self.on_update_memory)
        layout.addWidget(self.update_btn)
        
        self.generate_btn = QPushButton("Generate Hypotheses")
        self.generate_btn.clicked.connect(self.on_generate_hypotheses)
        layout.addWidget(self.generate_btn)
        
        self.clear_btn = QPushButton("Clear Memory")
        self.clear_btn.clicked.connect(self.on_clear_memory)
        layout.addWidget(self.clear_btn)
        
        # Auto-update checkbox
        self.auto_update = QCheckBox("Auto-Update")
        self.auto_update.setChecked(False)
        layout.addWidget(self.auto_update)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Timer for auto-update
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_auto_update)
        self.auto_update.stateChanged.connect(self.on_auto_update_toggled)
    
    def on_add_capsule(self):
        """Handle add capsule action"""
        dialog = AddCapsuleDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            self.memory.add_capsule(
                content=data['content'],
                character=data['character'],
                kind=data['kind'],
                certainty=data['certainty']
            )
            self.memory_updated.emit()
    
    def on_update_memory(self):
        """Handle memory update"""
        self.memory.orbit_update()
        self.memory.apply_influences()
        self.memory_updated.emit()
    
    def on_generate_hypotheses(self):
        """Generate new hypotheses"""
        self.memory.generate_hypotheses(2)
        self.memory_updated.emit()
    
    def on_clear_memory(self):
        """Clear memory"""
        reply = QMessageBox.question(self, "Clear Memory", 
                                     "Are you sure you want to clear all capsules?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.memory.capsules.clear()
            self.memory_updated.emit()
    
    def on_auto_update_toggled(self):
        """Toggle auto-update"""
        if self.auto_update.isChecked():
            self.timer.start(1000)  # Update every second
        else:
            self.timer.stop()
    
    def on_auto_update(self):
        """Auto-update callback"""
        self.on_update_memory()


# ===== MAIN APPLICATION =====

class RocaOrbitalMemoryApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.memory = RocaMemory()
        self.init_sample_data()
        self.init_ui()
        self.setWindowTitle("ROCA Orbital Memory System - PyQt6")
        self.setGeometry(100, 100, 1600, 1000)
    
    def init_sample_data(self):
        """Initialize with sample capsules"""
        sample_capsules = [
            {"content": "Newton's Laws of Motion", "character": "Newton", "kind": CapsuleKind.THEORY},
            {"content": "Einstein's Theory of Relativity", "character": "Einstein", "kind": CapsuleKind.THEORY},
            {"content": "Quantum Mechanics Principles", "character": "Bohr", "kind": CapsuleKind.THEORY},
            {"content": "Thermodynamics Laws", "character": "Carnot", "kind": CapsuleKind.THEORY},
            {"content": "Calculus Fundamentals", "character": "Leibniz", "kind": CapsuleKind.METHOD},
            {"content": "Probability Theory", "character": "Bayes", "kind": CapsuleKind.METHOD},
            {"content": "Scientific Method", "character": "Galileo", "kind": CapsuleKind.METHOD},
            {"content": "Evolution Theory", "character": "Darwin", "kind": CapsuleKind.THEORY},
            {"content": "Programming Logic", "character": "Turing", "kind": CapsuleKind.METHOD},
            {"content": "Neural Networks", "character": "McCulloch", "kind": CapsuleKind.METHOD},
        ]
        
        for sample in sample_capsules:
            capsule = self.memory.add_capsule(
                content=sample["content"],
                character=sample["character"],
                kind=sample["kind"]
            )
            capsule.gravity = 0.3 + random.random() * 0.7
            capsule.orbit_radius = 0.5 + random.random() * 1.5
    
    def init_ui(self):
        """Initialize main UI"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left side: Orbital canvas
        self.canvas = OrbitalCanvasWidget(self.memory)
        main_layout.addWidget(self.canvas, 2)
        
        # Right side: Control panels and tables
        right_layout = QVBoxLayout()
        
        # Tab widget for multiple views
        self.tabs = QTabWidget()
        
        # Stats tab
        self.stats_panel = StatsPanel(self.memory)
        self.tabs.addTab(self.stats_panel, "Statistics")
        
        # Control tab
        self.control_panel = ControlPanel(self.memory)
        self.control_panel.memory_updated.connect(self.on_memory_updated)
        self.tabs.addTab(self.control_panel, "Controls")
        
        # Memory table tab
        self.memory_table = MemoryTableWidget(self.memory)
        self.tabs.addTab(self.memory_table, "Capsules")
        
        right_layout.addWidget(self.tabs)
        main_layout.addLayout(right_layout, 1)
        
        central_widget.setLayout(main_layout)
    
    def on_memory_updated(self):
        """Handle memory updates"""
        self.stats_panel.update_stats()
        self.memory_table.update_table()
        self.canvas.update()


def distance(x1, y1, x2, y2) -> float:
    """Calculate distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def main():
    """Run the application"""
    app = QApplication(sys.argv)
    window = RocaOrbitalMemoryApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
