from random import randint

import numpy as np
import voluptuous as vol

from ledfx.color import RGB, parse_color, validate_color
from ledfx.color import Gradient, parse_gradient, validate_gradient

from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.droplets import DROPLET_NAMES, load_droplet

import logging
_LOGGER = logging.getLogger(__name__)

class RainAudioEffect(AudioReactiveEffect):
    NAME = "Rain Extend"
    CATEGORY = "Classic"

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional("mirror", description="Mirror the effect", default=True): bool,
            # TODO drops should be controlled by some sort of effectlet class,
            # which will provide a list of available drop names rather than just
            # this static range

            vol.Optional("pulse_strip", description="Pulse the entire strip to the beat", default="Off"): vol.In(["Off", "Lows", "Mids", "Highs"]),
            vol.Optional("pulse_only", description="Pulse will disable the drop", default=False): bool,
            vol.Optional("pulse_sensitivity", description="Sensitivity to pulse sounds", default=0.2): vol.All(vol.Coerce(float), vol.Range(min=0.03, max=0.4)),
            vol.Optional("pulse_decay", description="How fast the pulse should fade", default=0.5): vol.All(vol.Coerce(float), vol.Range(min=0.01, max=1.0)),

            vol.Optional("low_pulse_color", description="color for low pulses sounds, ie beats", default="white"): validate_color,
            vol.Optional("mid_pulse_color", description="color for mid pulses sounds, ie vocals", default="blue"): validate_color,
            vol.Optional("high_pulse_color", description="color for high pulses sounds, ie hi hat", default="purple"): validate_color,

            vol.Optional("low_sensitivity", description="Sensitivity to low sounds", default=0.2): vol.All(vol.Coerce(float), vol.Range(min=0.03, max=0.4)),
            vol.Optional("mid_sensitivity", description="Sensitivity to mid sounds", default=0.2): vol.All(vol.Coerce(float), vol.Range(min=0.03, max=0.4)),
            vol.Optional("high_sensitivity", description="Sensitivity to high sounds", default=0.2): vol.All(vol.Coerce(float), vol.Range(min=0.03, max=0.4)),

            vol.Optional("low_animation", description="Droplet animation style", default=DROPLET_NAMES[0]): vol.In(DROPLET_NAMES),
            vol.Optional("mid_animation", description="Droplet animation style", default=DROPLET_NAMES[0]): vol.In(DROPLET_NAMES),
            vol.Optional("high_animation", description="Droplet animation style", default=DROPLET_NAMES[0]): vol.In(DROPLET_NAMES),

            vol.Optional("low_gradient", description="Color gradient to display", default="red"): validate_gradient,
            vol.Optional("mid_gradient", description="Color gradient to display", default="green"): validate_gradient,
            vol.Optional("high_gradient", description="Color gradient to display", default="yellow"): validate_gradient,
        }
    )

    def on_activate(self, pixel_count):
        self.low_drop_frames = np.zeros(self.pixel_count, dtype=int)
        self.low_drop_colors = np.zeros((3, self.pixel_count))

        self.mid_drop_frames = np.zeros(self.pixel_count, dtype=int)
        self.mid_drop_colors = np.zeros((3, self.pixel_count))
        
        self.high_drop_frames = np.zeros(self.pixel_count, dtype=int)
        self.high_drop_colors = np.zeros((3, self.pixel_count))

        self.pulse_pixels = np.zeros((self.pixel_count, 3))

    def config_updated(self, config):
        self.low_animation = load_droplet(config["low_animation"])
        self.low_frames, self.low_frame_width = np.shape(self.low_animation)
        self.low_frame_centre_index = self.low_frame_width // 2
        self.low_frame_side_lengths = self.low_frame_centre_index - 1

        self.mid_animation = load_droplet(config["mid_animation"])
        self.mid_frames, self.mid_frame_width = np.shape(self.mid_animation)
        self.mid_frame_centre_index = self.mid_frame_width // 2
        self.mid_frame_side_lengths = self.mid_frame_centre_index - 1

        self.high_animation = load_droplet(config["high_animation"])
        self.high_frames, self.high_frame_width = np.shape(self.high_animation)
        self.high_frame_centre_index = self.high_frame_width // 2
        self.high_frame_side_lengths = self.high_frame_centre_index - 1

        self.intensity_filter = self.create_filter(alpha_decay=0.5, alpha_rise=0.99)
        self.filtered_intensities = np.zeros(3)

    def interpolate_gradient_color(self, gradient: Gradient, position: float) -> RGB:
        """
        Interpolates a color from a gradient at a given position.

        Args:
            gradient (Gradient): The gradient object.
            [(RGB(red=255, green=0, blue=0), 0.0), (RGB(red=255, green=255, blue=0), 0.35), (RGB(red=14, green=0, blue=255), 0.67), (RGB(red=0, green=255, blue=0), 1.0)]
            position (float): A value between 0 and 1 representing the position along the gradient.

        Returns:
            RGB: The interpolated RGB color.
        """
        colors, stops = zip(*gradient.colors)  # Extract colors and positions
        stops = np.array(stops)
        
        # Find where 'position' fits in the gradient
        idx = np.searchsorted(stops, position, side="right") - 1
        idx = np.clip(idx, 0, len(colors) - 2)  # Ensure index is within range

        # Get the two surrounding colors
        c1, c2 = colors[idx], colors[idx + 1]
        p1, p2 = stops[idx], stops[idx + 1]

        if (p1 == p2):
            factor = position
        else:
            factor = (position - p1) / (p2 - p1)
            
        # Linear interpolation
        interpolated_color = RGB(
            int(c1.red + factor * (c2.red - c1.red)),
            int(c1.green + factor * (c2.green - c1.green)),
            int(c1.blue + factor * (c2.blue - c1.blue))
        )
        
        return interpolated_color

    def new_drop(self, location, rgb_or_gradient, freq):
        gradient = parse_gradient(rgb_or_gradient)

        if (isinstance(gradient, Gradient)):
            position = np.random.rand()
            color = self.interpolate_gradient_color(gradient, position)
        else:
            color = gradient

        """
        Add a new drop animation
        TODO (?) this method overwrites a running drop animation in the same location
        would need a significant restructure to fix
        """
        if (freq == 0):
            self.low_drop_frames[location] = 1
            self.low_drop_colors[:, location] = color
        elif (freq == 1):
            self.mid_drop_frames[location] = 1
            self.mid_drop_colors[:, location] = color
        elif (freq == 2):
            self.high_drop_frames[location] = 1
            self.high_drop_colors[:, location] = color

    def update_drop_frames(self):
        # Set any drops at final frame back to 0 and remove color data
        finished_drops1 = self.low_drop_frames >= self.low_frames - 1
        self.low_drop_frames[finished_drops1] = 0
        self.low_drop_colors[:, finished_drops1] = 0
        # Add one to any running frames
        self.low_drop_frames[self.low_drop_frames > 0] += 1

        # Set any drops at final frame back to 0 and remove color data
        finished_drops2 = self.mid_drop_frames >= self.mid_frames - 1
        self.mid_drop_frames[finished_drops2] = 0
        self.mid_drop_colors[:, finished_drops2] = 0
        # Add one to any running frames
        self.mid_drop_frames[self.mid_drop_frames > 0] += 1

        # Set any drops at final frame back to 0 and remove color data
        finished_drops3 = self.high_drop_frames >= self.high_frames - 1
        self.high_drop_frames[finished_drops3] = 0
        self.high_drop_colors[:, finished_drops3] = 0
        # Add one to any running frames
        self.high_drop_frames[self.high_drop_frames > 0] += 1

    def render(self):
        """
        Get colored pixel data of all drops overlaid
        """
        # 2d array containing color intensity data
        low_overlaid_frames = np.zeros((3, self.pixel_count + self.low_frame_width))
        # Indexes of active drop animations
        low_drop_indices = np.flatnonzero(self.low_drop_frames)
        # TODO vectorize this to remove for loop
        for index in low_drop_indices:
            if self.low_drop_frames[index] >= len(self.low_animation):
                continue
            low_colored_frame = [self.low_animation[self.low_drop_frames[index]] * self.low_drop_colors[color, index] for color in range(3)]
            low_overlaid_frames[:, index : index + self.low_frame_width] += low_colored_frame

        mid_overlaid_frames = np.zeros((3, self.pixel_count + self.mid_frame_width))
        mid_drop_indices = np.flatnonzero(self.mid_drop_frames)
        for index in mid_drop_indices:
            if self.mid_drop_frames[index] >= len(self.mid_animation):
                continue
            mid_colored_frame = [self.mid_animation[self.mid_drop_frames[index]] * self.mid_drop_colors[color, index] for color in range(3)]
            mid_overlaid_frames[:, index : index + self.mid_frame_width] += mid_colored_frame

        high_overlaid_frames = np.zeros((3, self.pixel_count + self.high_frame_width))
        high_drop_indices = np.flatnonzero(self.high_drop_frames)
        for index in high_drop_indices:
            if self.high_drop_frames[index] >= len(self.high_animation):
                continue
            high_colored_frame = [self.high_animation[self.high_drop_frames[index]] * self.high_drop_colors[color, index] for color in range(3)]
            high_overlaid_frames[:, index : index + self.high_frame_width] += high_colored_frame

        self.pixels = low_overlaid_frames[:, self.low_frame_side_lengths : self.low_frame_side_lengths + self.pixel_count].T
        self.pixels += mid_overlaid_frames[:, self.mid_frame_side_lengths : self.mid_frame_side_lengths + self.pixel_count].T
        self.pixels += high_overlaid_frames[:, self.high_frame_side_lengths : self.high_frame_side_lengths + self.pixel_count].T
        self.pixels += self.pulse_pixels
        # Decay the pulse pixels

        factor = 6.0 + (9.99 - 6.0) * self._config["pulse_decay"]
        self.pulse_pixels = (self.pulse_pixels * factor) // 10
        #self.pulse_pixels = (self.pulse_pixels * 8)) // 10

    def strip_pulse(self, color):
        """
        Set the pulse pixels to a color
        This color decays over time in render()

        Args:
            color: The color to pulse the strip
        """
        self.pulse_pixels = np.array([color])

    def audio_data_updated(self, data):
        # Calculate the low, mids, and high indexes scaling based on the pixel
        # count
        intensities = np.fromiter((i.max() for i in self.melbank_thirds()), float)

        self.update_drop_frames()

        match self._config["pulse_strip"]:
            case "Lows":
                if (intensities[0] - self.filtered_intensities[0] > self._config["pulse_sensitivity"]):
                    self.strip_pulse(parse_color(self._config["low_pulse_color"]))
            case "Mids":
                if (intensities[1] - self.filtered_intensities[1] > self._config["pulse_sensitivity"]):
                    self.strip_pulse(parse_color(self._config["mid_pulse_color"]))
            case "Highs":
                if (intensities[2] - self.filtered_intensities[2] > self._config["pulse_sensitivity"]):
                    self.strip_pulse(parse_color(self._config["high_pulse_color"]))

        if (intensities[0] - self.filtered_intensities[0] > self._config["low_sensitivity"]):
            if (not (self._config["pulse_strip"] == "Lows" and self._config["pulse_only"])):
                self.new_drop(randint(0, self.pixel_count - 1), self._config["low_gradient"], 0)

        if (intensities[1] - self.filtered_intensities[1] > self._config["mid_sensitivity"]):
            if (not (self._config["pulse_strip"] == "Mids" and self._config["pulse_only"])):
                self.new_drop(randint(0, self.pixel_count - 1), self._config["mid_gradient"], 1)

        if (intensities[2] - self.filtered_intensities[2] > self._config["high_sensitivity"]):
            if (not (self._config["pulse_strip"] == "Highs" and self._config["pulse_only"])):
                self.new_drop(randint(0, self.pixel_count - 1), self._config["high_gradient"], 2)

        self.filtered_intensities = self.intensity_filter.update(intensities)
