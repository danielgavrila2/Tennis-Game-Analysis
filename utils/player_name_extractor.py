import re
from PIL import Image
import numpy as np
import pytesseract
import cv2
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


class TennisPlayerExtractor:
    def __init__(self):
        # Here you'll need to specify the executable
        pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

    def extract_scoreboard_region(self, image):
        """
        Extract the scoreboard region from the tennis match image.
        Typically located at the bottom of the screen.
        """
        height, width = image.shape[:2]

        # Define scoreboard region (bottom 15% of the image)
        scoreboard_y_start = int(height * 0.85)
        scoreboard_region = image[scoreboard_y_start:height, 0:width]

        return scoreboard_region

    def preprocess_image(self, image):
        """
        Preprocess the image for better OCR results.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold to get better contrast
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Scale up the image for better OCR
        scaled = cv2.resize(cleaned, None, fx=2, fy=2,
                            interpolation=cv2.INTER_CUBIC)

        return scaled

    def extract_player_names(self, image_path=r"D:\PROJECTS\Tennis Game Analysis\input_videos\image_captured.jpg"):
        """
        Extract player names from tennis match screenshot.
        """
        # Load image

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Extract scoreboard region
        scoreboard = self.extract_scoreboard_region(image)

        # Preprocess for OCR
        processed = self.preprocess_image(scoreboard)

        # Apply OCR
        text = pytesseract.image_to_string(
            processed, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ')

        # Clean and extract names
        player_names = self.parse_player_names(text)

        return player_names

    def parse_player_names(self, text):
        """
        Parse player names from extracted text.
        """
        # Clean the text
        text = text.strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        player_names = []

        # Common tennis player name patterns
        name_patterns = [
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # First Last
            r'[A-Z][a-z]+',  # Single name (like just surname)
            r'[A-Z]\.\s*[A-Z][a-z]+',  # A. Surname
        ]

        for line in lines:
            for pattern in name_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    # Filter out common false positives
                    if self.is_likely_player_name(match):
                        player_names.append(match.strip())

        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in player_names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)

        return unique_names[:2]  # Return max 2 players

    def is_likely_player_name(self, text):
        """
        Filter out text that's unlikely to be a player name.
        """
        # Skip very short or long strings
        if len(text) < 3 or len(text) > 30:
            return False

        # Skip common UI elements
        excluded_words = {
            'SET', 'GAME', 'MATCH', 'POINT', 'SCORE', 'TIME', 'FIRST', 'SECOND',
            'SERVE', 'ACE', 'FAULT', 'OUT', 'NET', 'WINNER', 'ERROR', 'ATP',
            'WTA', 'LIVE', 'HD', 'COURT', 'SURFACE', 'ROUND', 'TOURNAMENT'
        }

        if text.upper() in excluded_words:
            return False

        # Should contain at least one letter
        if not any(c.isalpha() for c in text):
            return False

        return True

    def extract_with_bounding_boxes(self, image_path):
        """
        Extract player names along with their bounding box coordinates.
        """
        image = cv2.imread(image_path)
        scoreboard = self.extract_scoreboard_region(image)
        processed = self.preprocess_image(scoreboard)

        # Get bounding box data
        data = pytesseract.image_to_data(
            processed, output_type=pytesseract.Output.DICT)

        player_info = []

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            confidence = data['conf'][i]

            # Only high confidence
            if confidence > 30 and self.is_likely_player_name(text):
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                # Adjust coordinates back to original image space
                height, width = image.shape[:2]
                scoreboard_y_start = int(height * 0.85)

                player_info.append({
                    'name': text,
                    # Scale back down
                    'bbox': (x//2, (y//2) + scoreboard_y_start, w//2, h//2),
                    'confidence': confidence
                })

        return player_info

# Example usage


def main():
    extractor = TennisPlayerExtractor()

    image_path = r"input_videos/image_captured.jpg"

    try:
        # Extract just the names
        player_names = extractor.extract_player_names(image_path)
        print("Detected players:", player_names)

        # Extract with bounding boxes for tracking
        player_info = extractor.extract_with_bounding_boxes(image_path)
        print("\nPlayer info with bounding boxes:")
        for info in player_info:
            print(
                f"Name: {info['name']}, BBox: {info['bbox']}, Confidence: {info['confidence']}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have pytesseract installed: pip install pytesseract")
        print("And Tesseract OCR installed on your system")


if __name__ == "__main__":
    main()
