#!/usr/bin/env python3
"""
ROCA Orbital Memory System - Enhanced PyQt6 Version

This enhanced version includes:
- Full capsule functionality from original
- Real-time statistics
- Document ingestion
- User interaction processing
- Hypothesis generation with personality
- Advanced visualization features
"""

import sys
import re
import math
import random
import time
import json
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import numpy as np

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget, QTableWidget,
    QTableWidgetItem, QComboBox, QDoubleSpinBox, QCheckBox, QDialog,
    QMessageBox, QProgressBar, QScrollArea, QFrame, QSplitter, QMenuBar,
    QMenu, QFileDialog, QInputDialog
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QAction

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    GPU_TYPE = "NVIDIA" if torch.cuda.is_available() else "APPLE" if hasattr(torch.backends, 'mps') else "CPU"
except:
    GPU_AVAILABLE = False
    GPU_TYPE = "CPU"


# ===== ENHANCED CAPSULE CLASSES =====

class CapsuleKind(Enum):
    """Types of knowledge capsules"""
    THEORY = "theory"
    METHOD = "method"
    OBSERVATION = "observation"
    CONCEPT = "concept"
    HYPOTHESIS = "hypothesis"


@dataclass
class Capsule:
    """Enhanced knowledge capsule with full metadata"""
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
    uuid: str = field(default_factory=lambda: str(random.randint(100000, 999999)))
    timestamp: float = field(default_factory=time.time)
    proven_by: Optional[str] = None
    pose: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = np.random.rand(128)
        self.confidence_history.append(self.certainty)
        self.pose = {'certainty': self.certainty}
    
    def update_confidence(self, new_confidence: float):
        """Update capsule confidence"""
        self.certainty = max(0, min(1, new_confidence))
        self.confidence_history.append(self.certainty)
        self.pose['certainty'] = self.certainty


@dataclass 
class PersonalityTrait:
    """Personality trait with history"""
    level: float = 0.5
    history: List[float] = field(default_factory=list)
    
    def update(self, new_level: float):
        self.level = max(0, min(1, new_level))
        self.history.append(self.level)


# ===== ENHANCED MEMORY SYSTEM =====

class EnhancedRocaMemory:
    """Enhanced ROCA memory system with full functionality"""
    
    def __init__(self):
        self.capsules: List[Capsule] = []
        self.temporal_order_counter = 0
        
        # Cayde personality system
        self.cayde_personality = {
            'obsession_with_invariants': PersonalityTrait(0.8),
            'preference_for_geometric_intuition': PersonalityTrait(0.9),
            'distrust_of_unnecessary_constants': PersonalityTrait(0.7),
            'intolerance_for_inconsistency': PersonalityTrait(0.85),
            'willingness_to_discard_cherished_models': PersonalityTrait(0.8),
            'openness_to_revolution': 0.9,
            'current_focus': 'geometric_unification',
            'personality_evolution': [],
            'disagreements_with_scientists': [],
            'abandoned_intuitions': [],
            'time_awareness': {'current_year': 1905}
        }
    
    def add_capsule(self, content: str, character: str = "unknown", 
                   kind: CapsuleKind = CapsuleKind.CONCEPT,
                   perspective: str = "default", success_status: Optional[str] = None,
                   proven_by: Optional[str] = None, certainty: float = 0.5, **kwargs) -> Capsule:
        """Add a new capsule to memory"""
        capsule = Capsule(
            content=content,
            character=character,
            kind=kind,
            perspective=perspective,
            success_status=success_status,
            proven_by=proven_by,
            certainty=certainty,
            temporal_order=self.temporal_order_counter,
            **kwargs
        )
        self.temporal_order_counter += 1
        self.capsules.append(capsule)
        return capsule
    
    def orbit_update(self):
        """Update orbital dynamics"""
        for capsule in self.capsules:
            # Orbital mechanics
            capsule.orbit_radius += random.uniform(-0.02, 0.02)
            capsule.gravity += random.uniform(-0.01, 0.01)
            
            # Clamp values
            capsule.gravity = max(0.1, min(1.0, capsule.gravity))
            capsule.orbit_radius = max(0.4, min(3.0, capsule.orbit_radius))
            
            # Confidence decay over time (simulation of forgetting)
            if capsule.certainty > 0.2:
                capsule.update_confidence(capsule.certainty * 0.98)
    
    def apply_gravitational_influence(self):
        """Apply gravitational influences between capsules"""
        for i, capsule1 in enumerate(self.capsules):
            for capsule2 in self.capsules[i+1:]:
                # Simple similarity-based attraction
                similarity = np.dot(capsule1.embedding, capsule2.embedding)
                
                if similarity > 0.7:  # Similar capsules attract
                    capsule1.gravity += similarity * 0.05
                    capsule2.gravity += similarity * 0.05
                elif similarity < 0.3:  # Dissimilar capsules repel
                    capsule1.gravity -= (1 - similarity) * 0.02
                    capsule2.gravity -= (1 - similarity) * 0.02
    
    def merge_similar(self, threshold: float = 0.8):
        """Merge similar capsules"""
        merged = []
        remaining = list(self.capsules)
        
        while remaining:
            primary = remaining.pop(0)
            similar = []
            
            for other in list(remaining):
                similarity = np.dot(primary.embedding, other.embedding)
                if similarity > threshold:
                    similar.append(other)
                    remaining.remove(other)
            
            if similar:
                # Merge capsules
                primary.certainty = max(c.certainty for c in [primary] + similar)
                primary.gravity = sum(c.gravity for c in [primary] + similar) / (len(similar) + 1)
                primary.insight_potential = max(c.insight_potential for c in [primary] + similar)
                merged.append(primary)
            else:
                merged.append(primary)
        
        self.capsules = merged
    
    def generate_hypotheses(self, num: int = 1) -> List[Capsule]:
        """Generate new hypotheses based on trends"""
        hypotheses = []
        
        if not self.capsules:
            return hypotheses
        
        for _ in range(num):
            # Select random capsules to base hypothesis on
            base_capsules = random.sample(self.capsules, min(3, len(self.capsules)))
            
            # Create hypothesis by combining insights
            combined_content = " + ".join(c.content[:20] for c in base_capsules)
            
            hypothesis = self.add_capsule(
                content=f"Hypothesis: Integration of {combined_content}...",
                character="cayde",
                kind=CapsuleKind.HYPOTHESIS,
                perspective="generated",
                certainty=0.4
            )
            hypothesis.insight_potential = 0.6 + random.random() * 0.3
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def get_core_capsules(self) -> List[Capsule]:
        """Get high-gravity core capsules"""
        return sorted(
            [c for c in self.capsules if c.gravity > 0.7],
            key=lambda c: c.gravity,
            reverse=True
        )
    
    def ingest_document(self, text: str, source: str = "document", 
                       author: Optional[str] = None) -> List[Capsule]:
        """Ingest text and create capsules"""
        capsules = []
        
        # Extract author if not provided
        if not author:
            author_match = re.search(r'by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text)
            if author_match:
                author = author_match.group(1)
            else:
                author = source
        
        # Split into sentences and create capsules
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only process meaningful sentences
                # Determine capsule kind
                kind = CapsuleKind.CONCEPT
                if any(w in sentence.lower() for w in ['theory', 'principle', 'law']):
                    kind = CapsuleKind.THEORY
                elif any(w in sentence.lower() for w in ['method', 'technique', 'process']):
                    kind = CapsuleKind.METHOD
                elif any(w in sentence.lower() for w in ['observed', 'measured', 'found']):
                    kind = CapsuleKind.OBSERVATION
                
                capsule = self.add_capsule(
                    content=sentence,
                    character=author,
                    kind=kind,
                    perspective="document",
                    certainty=0.6
                )
                capsules.append(capsule)
        
        return capsules
    
    def process_user_query(self, query: str) -> Dict[str, Any]:
        """Process user query and generate response"""
        response = {
            'query': query,
            'relevant_capsules': [],
            'generated_insights': [],
            'response': ""
        }
        
        # Find relevant capsules
        query_words = set(query.lower().split())
        for capsule in self.capsules:
            capsule_words = set(capsule.content.lower().split())
            overlap = len(query_words.intersection(capsule_words))
            if overlap > 0:
                response['relevant_capsules'].append({
                    'capsule': capsule,
                    'relevance': overlap / len(query_words)
                })
        
        # Sort by relevance
        response['relevant_capsules'].sort(key=lambda x: x['relevance'], reverse=True)
        
        # Generate response
        if response['relevant_capsules']:
            best = response['relevant_capsules'][0]['capsule']
            response['response'] = f"Based on my knowledge: {best.content}"
        else:
            response['response'] = "I don't have information about that. Could you tell me more?"
        
        return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'total_capsules': len(self.capsules),
            'core_capsules': len(self.get_core_capsules()),
            'by_kind': {
                kind.value: sum(1 for c in self.capsules if c.kind == kind)
                for kind in CapsuleKind
            },
            'by_character': dict(sorted(
                {c.character: sum(1 for x in self.capsules if x.character == c.character)
                 for c in self.capsules}.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),  # Top 10 characters
            'avg_certainty': np.mean([c.certainty for c in self.capsules]) if self.capsules else 0,
            'avg_gravity': np.mean([c.gravity for c in self.capsules]) if self.capsules else 0,
            'high_insight': sum(1 for c in self.capsules if c.insight_potential > 0.7)
        }


# ===== PyQt6 WIDGETS =====

class OrbitalCanvasWidget(QWidget):
    """Enhanced orbital visualization widget"""
    
    def __init__(self, memory: EnhancedRocaMemory):
        super().__init__()
        self.memory = memory
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #141e1e;")
        
        self.animation_time = 0
        self.selected_capsule: Optional[Capsule] = None
        self.zoom = 1.0
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)
    
    def animate(self):
        """Update animation"""
        self.animation_time += 0.016
        self.update()
    
    def paintEvent(self, event):
        """Paint orbital visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(20, 20, 30))
        
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # Draw orbital rings
        pen = QPen(QColor(60, 60, 80))
        pen.setWidth(1)
        painter.setPen(pen)
        
        orbit_radii = [80, 120, 160, 200, 240, 280]
        for radius in orbit_radii:
            painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
        
        # Draw core
        painter.setBrush(QBrush(QColor(100, 200, 255)))
        painter.drawEllipse(center_x - 25, center_y - 25, 50, 50)
        
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(center_x - 40, center_y + 50, 80, 20,
                        Qt.AlignmentFlag.AlignCenter, "ROCA Core")
        
        # Draw capsules
        if self.memory.capsules:
            capsules_per_orbit = [8, 12, 16, 20, 24, 28]
            
            capsule_index = 0
            for orbit_level, (radius, max_per_orbit) in enumerate(zip(orbit_radii, capsules_per_orbit)):
                orbit_capsules = self.memory.capsules[capsule_index:capsule_index + max_per_orbit]
                capsule_index += max_per_orbit
                
                if not orbit_capsules:
                    continue
                
                for i, capsule in enumerate(orbit_capsules):
                    angle = (i / len(orbit_capsules)) * 2 * math.pi + \
                           self.animation_time * 0.1 * (orbit_level + 1) * 0.1
                    
                    x = center_x + math.cos(angle) * radius
                    y = center_y + math.sin(angle) * radius
                    
                    # Determine color
                    if capsule.character == "cayde":
                        color = QColor(0, 255, 255)
                    elif capsule.insight_potential > 0.7:
                        color = QColor(255, 255, 0)
                    elif capsule.success_status == "proven_later":
                        color = QColor(255, 0, 255)
                    else:
                        color = QColor(
                            int(100 + capsule.certainty * 155),
                            int(100 + capsule.gravity * 100),
                            int(150 + capsule.orbit_radius * 50)
                        )
                    
                    # Draw capsule
                    capsule_radius = 12
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.drawEllipse(int(x) - capsule_radius, int(y) - capsule_radius,
                                       capsule_radius * 2, capsule_radius * 2)
                    
                    # Draw label
                    name = str(capsule.character)[:7]
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(QFont("Arial", 8))
                    painter.drawText(int(x) - 20, int(y) + capsule_radius + 15, 40, 20,
                                    Qt.AlignmentFlag.AlignCenter, name)
    
    def mousePressEvent(self, event):
        """Handle mouse clicks"""
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        click_x = event.position().x()
        click_y = event.position().y()
        
        # Simple capsule selection
        for capsule in self.memory.capsules:
            # You could add more sophisticated hit detection here
            pass
        
        self.update()


class MemoryTableWidget(QWidget):
    """Table view of memory capsules"""
    
    def __init__(self, memory: EnhancedRocaMemory):
        super().__init__()
        self.memory = memory
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Content", "Character", "Kind", "Certainty", "Gravity", "Status"
        ])
        layout.addWidget(self.table)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.update_table)
        layout.addWidget(refresh_btn)
        
        self.setLayout(layout)
        self.update_table()
    
    def update_table(self):
        """Update table"""
        self.table.setRowCount(len(self.memory.capsules))
        
        for row, capsule in enumerate(self.memory.capsules):
            content = (capsule.content[:50] + "...") if len(capsule.content) > 50 else capsule.content
            self.table.setItem(row, 0, QTableWidgetItem(content))
            self.table.setItem(row, 1, QTableWidgetItem(capsule.character))
            self.table.setItem(row, 2, QTableWidgetItem(capsule.kind.value))
            self.table.setItem(row, 3, QTableWidgetItem(f"{capsule.certainty:.2f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{capsule.gravity:.2f}"))
            status = capsule.success_status or "unknown"
            self.table.setItem(row, 5, QTableWidgetItem(status))


class StatisticsWidget(QWidget):
    """Statistics display widget"""
    
    def __init__(self, memory: EnhancedRocaMemory):
        super().__init__()
        self.memory = memory
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Memory Statistics")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Stats labels
        self.stats_labels = {}
        for key in ['total_capsules', 'core_capsules', 'avg_certainty', 'avg_gravity', 'high_insight']:
            label = QLabel()
            self.stats_labels[key] = label
            layout.addWidget(label)
        
        # GPU status
        gpu_label = QLabel(f"GPU: {GPU_TYPE} {'✓' if GPU_AVAILABLE else '✗'}")
        gpu_label.setStyleSheet(f"color: {'green' if GPU_AVAILABLE else 'red'};")
        layout.addWidget(gpu_label)
        
        layout.addStretch()
        self.setLayout(layout)
        self.update_stats()
    
    def update_stats(self):
        """Update statistics"""
        stats = self.memory.get_statistics()
        
        self.stats_labels['total_capsules'].setText(f"Total Capsules: {stats['total_capsules']}")
        self.stats_labels['core_capsules'].setText(f"Core Capsules: {stats['core_capsules']}")
        self.stats_labels['avg_certainty'].setText(f"Avg Certainty: {stats['avg_certainty']:.2f}")
        self.stats_labels['avg_gravity'].setText(f"Avg Gravity: {stats['avg_gravity']:.2f}")
        self.stats_labels['high_insight'].setText(f"High Insight: {stats['high_insight']}")


class ControlWidget(QWidget):
    """Control panel widget"""
    
    memory_updated = pyqtSignal()
    
    def __init__(self, memory: EnhancedRocaMemory):
        super().__init__()
        self.memory = memory
        self.init_ui()
    
    def init_ui(self):
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
        self.update_btn.clicked.connect(self.on_update)
        layout.addWidget(self.update_btn)
        
        self.generate_btn = QPushButton("Generate Hypotheses")
        self.generate_btn.clicked.connect(self.on_generate)
        layout.addWidget(self.generate_btn)
        
        self.ingest_btn = QPushButton("Ingest Document")
        self.ingest_btn.clicked.connect(self.on_ingest)
        layout.addWidget(self.ingest_btn)
        
        self.query_btn = QPushButton("Query Memory")
        self.query_btn.clicked.connect(self.on_query)
        layout.addWidget(self.query_btn)
        
        self.clear_btn = QPushButton("Clear Memory")
        self.clear_btn.clicked.connect(self.on_clear)
        layout.addWidget(self.clear_btn)
        
        # Auto-update
        self.auto_update = QCheckBox("Auto-Update")
        layout.addWidget(self.auto_update)
        
        layout.addStretch()
        self.setLayout(layout)
        
        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_auto_update)
        self.auto_update.stateChanged.connect(self.on_auto_toggled)
    
    def on_add_capsule(self):
        """Add a capsule"""
        dialog = AddCapsuleDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            data = dialog.get_data()
            self.memory.add_capsule(**data)
            self.memory_updated.emit()
    
    def on_update(self):
        """Update memory"""
        self.memory.orbit_update()
        self.memory.apply_gravitational_influence()
        self.memory_updated.emit()
    
    def on_generate(self):
        """Generate hypotheses"""
        self.memory.generate_hypotheses(2)
        self.memory_updated.emit()
    
    def on_ingest(self):
        """Ingest document"""
        dialog = IngestDocumentDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            text, author = dialog.get_data()
            self.memory.ingest_document(text, author=author)
            self.memory_updated.emit()
    
    def on_query(self):
        """Query memory"""
        query, ok = QInputDialog.getText(self, "Query Memory", "Enter your query:")
        if ok and query:
            result = self.memory.process_user_query(query)
            response_msg = result['response']
            if result['relevant_capsules']:
                response_msg += f"\n\nFound {len(result['relevant_capsules'])} relevant capsules."
            QMessageBox.information(self, "Query Result", response_msg)
    
    def on_clear(self):
        """Clear memory"""
        reply = QMessageBox.question(
            self, "Clear Memory", "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.memory.capsules.clear()
            self.memory_updated.emit()
    
    def on_auto_toggled(self):
        if self.auto_update.isChecked():
            self.timer.start(1000)
        else:
            self.timer.stop()
    
    def on_auto_update(self):
        self.on_update()


class AddCapsuleDialog(QDialog):
    """Dialog for adding capsules"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Content
        layout.addWidget(QLabel("Content:"))
        self.content = QTextEdit()
        self.content.setMaximumHeight(100)
        layout.addWidget(self.content)
        
        # Character
        layout.addWidget(QLabel("Character:"))
        self.character = QLineEdit()
        layout.addWidget(self.character)
        
        # Kind
        layout.addWidget(QLabel("Kind:"))
        self.kind = QComboBox()
        self.kind.addItems([k.value for k in CapsuleKind])
        layout.addWidget(self.kind)
        
        # Certainty
        layout.addWidget(QLabel("Certainty:"))
        self.certainty = QDoubleSpinBox()
        self.certainty.setRange(0, 1)
        self.certainty.setSingleStep(0.1)
        self.certainty.setValue(0.5)
        layout.addWidget(self.certainty)
        
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
        self.setWindowTitle("Add Capsule")
        self.setGeometry(100, 100, 500, 400)
    
    def get_data(self) -> dict:
        return {
            'content': self.content.toPlainText(),
            'character': self.character.text() or "unknown",
            'kind': CapsuleKind(self.kind.currentText()),
            'certainty': self.certainty.value()
        }


class IngestDocumentDialog(QDialog):
    """Dialog for ingesting documents"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Document Text:"))
        self.text = QTextEdit()
        layout.addWidget(self.text)
        
        layout.addWidget(QLabel("Author (optional):"))
        self.author = QLineEdit()
        layout.addWidget(self.author)
        
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("Ingest")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.setWindowTitle("Ingest Document")
        self.setGeometry(100, 100, 600, 500)
    
    def get_data(self) -> tuple:
        return (self.text.toPlainText(), self.author.text() or "document")


# ===== MAIN APPLICATION =====

class RocaOrbitalMemoryApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.memory = EnhancedRocaMemory()
        self.init_sample_data()
        self.init_ui()
        self.setup_menus()
        
        self.setWindowTitle("ROCA Orbital Memory System - PyQt6 Enhanced")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Auto-update timer
        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self.on_auto_update)
    
    def init_sample_data(self):
        """Initialize with sample data"""
        samples = [
            ("Newton's Laws of Motion", "Newton", CapsuleKind.THEORY),
            ("Einstein's Relativity", "Einstein", CapsuleKind.THEORY),
            ("Quantum Mechanics", "Bohr", CapsuleKind.THEORY),
            ("Thermodynamics", "Carnot", CapsuleKind.THEORY),
            ("Calculus", "Leibniz", CapsuleKind.METHOD),
            ("Probability Theory", "Bayes", CapsuleKind.METHOD),
            ("Evolution Theory", "Darwin", CapsuleKind.THEORY),
            ("Programming", "Turing", CapsuleKind.METHOD),
            ("Neural Networks", "McCulloch", CapsuleKind.METHOD),
            ("Machine Learning", "Minsky", CapsuleKind.METHOD),
        ]
        
        for content, character, kind in samples:
            c = self.memory.add_capsule(
                content=content,
                character=character,
                kind=kind
            )
            c.gravity = 0.3 + random.random() * 0.7
            c.orbit_radius = 0.5 + random.random() * 1.5
    
    def init_ui(self):
        """Initialize main UI"""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout()
        
        # Left: Canvas
        self.canvas = OrbitalCanvasWidget(self.memory)
        main_layout.addWidget(self.canvas, 2)
        
        # Right: Tabs
        self.tabs = QTabWidget()
        
        self.stats = StatisticsWidget(self.memory)
        self.tabs.addTab(self.stats, "Statistics")
        
        self.control = ControlWidget(self.memory)
        self.control.memory_updated.connect(self.on_memory_updated)
        self.tabs.addTab(self.control, "Controls")
        
        self.table = MemoryTableWidget(self.memory)
        self.tabs.addTab(self.table, "Capsules")
        
        main_layout.addWidget(self.tabs, 1)
        central.setLayout(main_layout)
    
    def setup_menus(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Exit", self.close)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Clear All", self.memory.capsules.clear)
        
        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Auto-Update", self.toggle_auto_update)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about)
    
    def toggle_auto_update(self):
        """Toggle auto-update"""
        if self.auto_timer.isActive():
            self.auto_timer.stop()
        else:
            self.auto_timer.start(1000)
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.information(
            self,
            "About ROCA Orbital",
            "ROCA Orbital Memory System - PyQt6 Enhanced\n\n"
            "A modern knowledge visualization and management system.\n\n"
            f"GPU: {GPU_TYPE}"
        )
    
    def on_memory_updated(self):
        """Handle memory updates"""
        self.stats.update_stats()
        self.table.update_table()
        self.canvas.update()
    
    def on_auto_update(self):
        """Auto-update callback"""
        self.memory.orbit_update()
        self.memory.apply_gravitational_influence()
        self.on_memory_updated()


def main():
    """Run the application"""
    app = QApplication(sys.argv)
    window = RocaOrbitalMemoryApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
