#!/usr/bin/env python3
"""
ROCA Audio Waveform Visualization - Waveform Display and Audio-Video Sync

Advanced audio waveform visualization for precise audio-visual synchronization.
Provides scalable, clickable waveform overlay with beat detection and timeline scrubbing.

Features:
- Real-time waveform rendering with Pygame
- Zoom in/out functionality
- Click-to-scrub timeline control
- Beat detection and visual markers
- Waveform data caching in AudioCapsule
- Audio-visual synchronization
"""

import numpy as np
import pygame
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import time
import uuid


@dataclass
class AudioWaveformData:
    """Container for audio waveform data and metadata"""
    samples: np.ndarray  # Raw audio samples
    sample_rate: int     # Sample rate in Hz
    duration: float      # Duration in seconds
    channels: int        # Number of audio channels
    waveform_peaks: Optional[np.ndarray] = None  # Pre-computed peak values for visualization
    beat_positions: List[float] = field(default_factory=list)  # Beat positions in seconds
    amplitude_envelope: np.ndarray = None  # RMS amplitude envelope

    def __post_init__(self):
        if self.amplitude_envelope is None and len(self.samples) > 0:
            self.compute_amplitude_envelope()

    def compute_amplitude_envelope(self, window_size: int = 1024):
        """Compute RMS amplitude envelope for visualization"""
        if len(self.samples) == 0:
            self.amplitude_envelope = np.array([])
            return

        # Convert to mono if stereo
        if self.channels == 2:
            mono_samples = (self.samples[:, 0] + self.samples[:, 1]) / 2
        else:
            mono_samples = self.samples.flatten()

        # Compute RMS in windows
        num_windows = len(mono_samples) // window_size
        envelope = np.zeros(num_windows)

        for i in range(num_windows):
            start = i * window_size
            end = min(start + window_size, len(mono_samples))
            window = mono_samples[start:end]
            envelope[i] = np.sqrt(np.mean(window ** 2))

        # Normalize to 0-1 range
        if np.max(envelope) > 0:
            envelope = envelope / np.max(envelope)

        self.amplitude_envelope = envelope

    def detect_beats(self, threshold: float = 0.3, min_interval: float = 0.3):
        """Simple beat detection based on amplitude envelope"""
        if self.amplitude_envelope is None or len(self.amplitude_envelope) == 0:
            return

        self.beat_positions = []
        last_beat_time = -min_interval

        # Convert envelope indices to time
        time_per_window = len(self.samples) / (self.sample_rate * len(self.amplitude_envelope))

        for i, amplitude in enumerate(self.amplitude_envelope):
            current_time = i * time_per_window

            if amplitude > threshold and (current_time - last_beat_time) > min_interval:
                self.beat_positions.append(current_time)
                last_beat_time = current_time


@dataclass
class WaveformDisplaySettings:
    """Settings for waveform display"""
    height: int = 100  # Height of waveform display in pixels
    color: Tuple[int, int, int] = (100, 200, 255)  # Waveform color (RGB)
    background_color: Tuple[int, int, int] = (30, 30, 40)  # Background color
    beat_marker_color: Tuple[int, int, int] = (255, 100, 100)  # Beat marker color
    playhead_color: Tuple[int, int, int] = (255, 255, 100)  # Playhead color
    zoom_level: float = 1.0  # Zoom level (1.0 = normal, >1.0 = zoomed in)
    show_beats: bool = True  # Whether to show beat markers
    show_playhead: bool = True  # Whether to show playhead


class AudioWaveformWidget:
    """
    Pygame-based audio waveform visualization widget

    Provides scalable, clickable waveform display with beat detection
    and timeline scrubbing capabilities.
    """

    def __init__(self, x: int, y: int, width: int, waveform_data: Optional[AudioWaveformData] = None):
        self.x = x
        self.y = y
        self.width = width
        self.waveform_data = waveform_data

        # Display settings
        self.settings = WaveformDisplaySettings()

        # Interaction state
        self.is_dragging = False
        self.drag_start_x = 0
        self.current_playhead_time = 0.0  # Current playhead position in seconds
        self.zoom_center_time = 0.0  # Center time for zoom operations

        # Cached waveform peaks for performance
        self.cached_peaks = None
        self.cache_valid = False

        # Callbacks
        self.on_playhead_changed = None  # Callback for playhead changes
        self.on_zoom_changed = None      # Callback for zoom changes

    def set_waveform_data(self, waveform_data: AudioWaveformData):
        """Set the waveform data to display"""
        self.waveform_data = waveform_data
        self.cache_valid = False
        self.current_playhead_time = 0.0
        self.zoom_center_time = 0.0

    def set_playhead_time(self, time_seconds: float):
        """Set the current playhead position"""
        if self.waveform_data:
            self.current_playhead_time = max(0.0, min(time_seconds, self.waveform_data.duration))
        else:
            self.current_playhead_time = time_seconds

    def get_playhead_time(self) -> float:
        """Get the current playhead position"""
        return self.current_playhead_time

    def set_zoom_level(self, zoom: float, center_time: Optional[float] = None):
        """Set the zoom level and optional center time"""
        self.settings.zoom_level = max(0.1, min(zoom, 10.0))
        if center_time is not None:
            self.zoom_center_time = center_time
        self.cache_valid = False

        if self.on_zoom_changed:
            self.on_zoom_changed(self.settings.zoom_level)

    def zoom_in(self, factor: float = 1.5):
        """Zoom in by the specified factor"""
        center_time = self._screen_x_to_time(self.width // 2)
        self.set_zoom_level(self.settings.zoom_level * factor, center_time)

    def zoom_out(self, factor: float = 1.5):
        """Zoom out by the specified factor"""
        center_time = self._screen_x_to_time(self.width // 2)
        self.set_zoom_level(self.settings.zoom_level / factor, center_time)

    def _time_to_screen_x(self, time_seconds: float) -> int:
        """Convert time to screen X coordinate"""
        if not self.waveform_data or self.waveform_data.duration == 0:
            return 0

        # Calculate visible time range based on zoom
        total_visible_duration = self.waveform_data.duration / self.settings.zoom_level
        start_time = max(0, self.zoom_center_time - total_visible_duration / 2)
        end_time = min(self.waveform_data.duration, start_time + total_visible_duration)

        if end_time <= start_time:
            return 0

        # Convert time to screen coordinate
        relative_time = (time_seconds - start_time) / (end_time - start_time)
        return int(self.x + relative_time * self.width)

    def _screen_x_to_time(self, screen_x: int) -> float:
        """Convert screen X coordinate to time"""
        if not self.waveform_data or self.waveform_data.duration == 0:
            return 0.0

        # Calculate visible time range
        total_visible_duration = self.waveform_data.duration / self.settings.zoom_level
        start_time = max(0, self.zoom_center_time - total_visible_duration / 2)

        # Convert screen coordinate to time
        relative_x = (screen_x - self.x) / self.width
        return start_time + relative_x * total_visible_duration

    def _compute_waveform_peaks(self):
        """Compute or cache waveform peaks for the current zoom level"""
        if not self.waveform_data or not self.waveform_data.amplitude_envelope.size:
            self.cached_peaks = np.array([])
            return

        # Calculate visible time range
        total_visible_duration = self.waveform_data.duration / self.settings.zoom_level
        start_time = max(0, self.zoom_center_time - total_visible_duration / 2)
        end_time = min(self.waveform_data.duration, start_time + total_visible_duration)

        # Convert time range to envelope indices
        time_per_window = self.waveform_data.duration / len(self.waveform_data.amplitude_envelope)
        start_idx = int(start_time / time_per_window)
        end_idx = int(end_time / time_per_window)

        # Get visible envelope data
        visible_envelope = self.waveform_data.amplitude_envelope[
            max(0, start_idx):min(len(self.waveform_data.amplitude_envelope), end_idx)
        ]

        if len(visible_envelope) == 0:
            self.cached_peaks = np.array([])
            return

        # Downsample or upsample to match display width
        if len(visible_envelope) > self.width:
            # Downsample by taking max in windows
            window_size = len(visible_envelope) // self.width
            peaks = np.zeros(self.width)
            for i in range(self.width):
                start = i * window_size
                end = min(start + window_size, len(visible_envelope))
                peaks[i] = np.max(visible_envelope[start:end])
        else:
            # Upsample by interpolation
            peaks = np.interp(
                np.linspace(0, len(visible_envelope) - 1, self.width),
                np.arange(len(visible_envelope)),
                visible_envelope
            )

        self.cached_peaks = peaks
        self.cache_valid = True

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events for interaction"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()

            # Check if click is within waveform area
            if (self.x <= mouse_x <= self.x + self.width and
                self.y <= mouse_y <= self.y + self.settings.height):

                if event.button == 1:  # Left click - set playhead
                    new_time = self._screen_x_to_time(mouse_x)
                    self.set_playhead_time(new_time)
                    if self.on_playhead_changed:
                        self.on_playhead_changed(new_time)
                    return True

                elif event.button == 3:  # Right click - center zoom on this point
                    center_time = self._screen_x_to_time(mouse_x)
                    self.zoom_center_time = center_time
                    self.cache_valid = False
                    return True

                elif event.button == 4:  # Mouse wheel up - zoom in
                    self.zoom_in(1.2)
                    return True

                elif event.button == 5:  # Mouse wheel down - zoom out
                    self.zoom_out(1.2)
                    return True

        return False

    def draw(self, surface: pygame.Surface):
        """Draw the waveform on the given surface"""
        if not self.waveform_data:
            return

        # Update cache if needed
        if not self.cache_valid:
            self._compute_waveform_peaks()

        # Draw background
        pygame.draw.rect(surface, self.settings.background_color,
                        (self.x, self.y, self.width, self.settings.height))

        # Draw waveform
        if len(self.cached_peaks) > 0:
            center_y = self.y + self.settings.height // 2

            # Draw waveform as a filled shape
            points = [(self.x, center_y)]
            for i, peak in enumerate(self.cached_peaks):
                x = self.x + i
                y_top = center_y - int(peak * self.settings.height * 0.4)
                y_bottom = center_y + int(peak * self.settings.height * 0.4)
                points.extend([(x, y_top), (x, y_bottom)])
            points.append((self.x + self.width, center_y))

            if len(points) > 2:
                pygame.draw.polygon(surface, self.settings.color, points, 1)

        # Draw beat markers
        if self.settings.show_beats and self.waveform_data.beat_positions:
            for beat_time in self.waveform_data.beat_positions:
                beat_x = self._time_to_screen_x(beat_time)
                if self.x <= beat_x <= self.x + self.width:
                    pygame.draw.line(surface, self.settings.beat_marker_color,
                                   (beat_x, self.y), (beat_x, self.y + self.settings.height), 2)

        # Draw playhead
        if self.settings.show_playhead:
            playhead_x = self._time_to_screen_x(self.current_playhead_time)
            if self.x <= playhead_x <= self.x + self.width:
                pygame.draw.line(surface, self.settings.playhead_color,
                               (playhead_x, self.y), (playhead_x, self.y + self.settings.height), 3)


class AudioCapsule:
    """Enhanced capsule for audio content with waveform data"""

    def __init__(self, audio_path: str, content: str = "", **kwargs):
        # Initialize as regular capsule - we'll need to import this dynamically
        try:
            # Try to import from the main module
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))

            from Roca_orbital import Capsule, CapsuleKind
            # Create a capsule instance
            capsule = Capsule(content=content, kind=CapsuleKind.AUDIO, **kwargs)
            # Copy all attributes to self
            self.__dict__.update(capsule.__dict__)
        except (ImportError, AttributeError):
            # Fallback if import fails - create basic attributes
            self.content = content
            self.kind = "audio"  # fallback string
            self.uuid = str(uuid.uuid4())
            self.embedding = None
            self.perspective = kwargs.get('perspective', "user")
            self.certainty = kwargs.get('certainty', 0.5)
            self.relevance = kwargs.get('relevance', "general")
            self.character = kwargs.get('character', None)
            self.persona = kwargs.get('persona', None)
            self.pose = kwargs.get('pose', {
                "temporal": None,
                "perspective": "user",
                "certainty": 0.5,
                "attention": 0.5,
                "relevance": 0.5,
                "abstraction": 0.5
            })
            self.gravity = kwargs.get('gravity', 0.5)
            self.orbit_radius = kwargs.get('orbit_radius', 1.0)
            self.links = kwargs.get('links', [])
            self.locked = kwargs.get('locked', False)
            self.parent = kwargs.get('parent', None)
            self.temporal_order = kwargs.get('temporal_order', int(time.time() * 1000))
            self.confidence_history = kwargs.get('confidence_history', [])
            self.paradigm_shifts = kwargs.get('paradigm_shifts', [])
            self.intellectual_conflicts = kwargs.get('intellectual_conflicts', [])
            self.success_status = kwargs.get('success_status', None)
            self.proven_by = kwargs.get('proven_by', None)
            self.insight_potential = kwargs.get('insight_potential', 0.0)

        self.audio_path = audio_path
        self.waveform_data: Optional[AudioWaveformData] = None
        self.waveform_widget: Optional[AudioWaveformWidget] = None

        self.audio_path = audio_path
        self.waveform_data: Optional[AudioWaveformData] = None
        self.waveform_widget: Optional[AudioWaveformWidget] = None

    def load_waveform_data(self):
        """Load and process audio waveform data"""
        try:
            # This would typically use a library like librosa or pydub
            # For now, we'll create mock data
            self._generate_mock_waveform_data()
        except Exception as e:
            print(f"Failed to load waveform data: {e}")

    def _generate_mock_waveform_data(self):
        """Generate mock waveform data for demonstration"""
        # Create synthetic audio data
        duration = 10.0  # 10 seconds
        sample_rate = 44100
        num_samples = int(duration * sample_rate)

        # Generate a mix of sine waves and noise to simulate music
        t = np.linspace(0, duration, num_samples)
        frequency = 440  # A4 note
        audio_signal = (
            0.5 * np.sin(2 * np.pi * frequency * t) +  # Fundamental
            0.3 * np.sin(2 * np.pi * 2 * frequency * t) +  # Octave
            0.2 * np.sin(2 * np.pi * 3 * frequency * t) +  # Fifth
            0.1 * np.random.normal(0, 0.1, num_samples)  # Noise
        )

        # Create stereo by duplicating
        stereo_audio = np.column_stack([audio_signal, audio_signal])

        self.waveform_data = AudioWaveformData(
            samples=stereo_audio,
            sample_rate=sample_rate,
            duration=duration,
            channels=2
        )

        # Detect beats (simple periodic beats for demo)
        beat_interval = 0.5  # 120 BPM
        self.waveform_data.beat_positions = [
            i * beat_interval for i in range(int(duration / beat_interval))
        ]

    def create_waveform_widget(self, x: int, y: int, width: int) -> AudioWaveformWidget:
        """Create and return a waveform widget for this audio capsule"""
        if not self.waveform_data:
            self.load_waveform_data()

        self.waveform_widget = AudioWaveformWidget(x, y, width, self.waveform_data)
        return self.waveform_widget


# Utility functions for audio processing
def load_audio_waveform(audio_path: str) -> Optional[AudioWaveformData]:
    """Load audio file and extract waveform data"""
    try:
        # This would use librosa, pydub, or similar
        # For now, return mock data
        waveform_data = AudioWaveformData(
            samples=np.random.normal(0, 0.1, (441000, 2)),  # 10 seconds of stereo audio
            sample_rate=44100,
            duration=10.0,
            channels=2
        )
        waveform_data.detect_beats()
        return waveform_data
    except Exception as e:
        print(f"Failed to load audio: {e}")
        return None


def create_audio_capsule_from_file(audio_path: str, content: str = "") -> AudioCapsule:
    """Create an AudioCapsule from an audio file"""
    capsule = AudioCapsule(audio_path, content)
    capsule.load_waveform_data()
    return capsule


# Test function
def test_waveform_visualization():
    """Test the waveform visualization functionality"""
    print("Testing Audio Waveform Visualization...")

    # Create mock audio capsule
    audio_capsule = AudioCapsule("test_audio.wav", "Test audio content")
    audio_capsule.load_waveform_data()

    print(f"Waveform data loaded: {audio_capsule.waveform_data.duration}s duration")
    print(f"Beat positions: {len(audio_capsule.waveform_data.beat_positions)} beats detected")

    # Create waveform widget
    widget = audio_capsule.create_waveform_widget(100, 100, 800)
    print(f"Waveform widget created at ({widget.x}, {widget.y}) with width {widget.width}")

    # Test zoom functionality
    widget.zoom_in(2.0)
    print(f"Zoomed in: zoom level = {widget.settings.zoom_level}")

    widget.zoom_out(2.0)
    print(f"Zoomed out: zoom level = {widget.settings.zoom_level}")

    print("Audio waveform visualization test completed!")


if __name__ == "__main__":
    test_waveform_visualization()