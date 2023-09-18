from pathlib import Path
from matplotlib.image import imread, imsave
import random

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


class Img:

    def __init__(self, path):
        """
        Do not change the constructor implementation
        """
        self.path = Path(path)
        self.data = rgb2gray(imread(path)).tolist()

    def save_img(self):
        """
        Do not change the below implementation
        """
        new_path = self.path.with_name(self.path.stem + '_filtered' + self.path.suffix)
        imsave(new_path, self.data, cmap='gray')
        return new_path

    def blur(self, blur_level=16):

        height = len(self.data)
        width = len(self.data[0])
        filter_sum = blur_level ** 2

        result = []
        for i in range(height - blur_level + 1):
            row_result = []
            for j in range(width - blur_level + 1):
                sub_matrix = [row[j:j + blur_level] for row in self.data[i:i + blur_level]]
                average = sum(sum(sub_row) for sub_row in sub_matrix) // filter_sum
                row_result.append(average)
            result.append(row_result)

        self.data = result

    def contour(self):
        for i, row in enumerate(self.data):
            res = []
            for j in range(1, len(row)):
                res.append(abs(row[j-1] - row[j]))

            self.data[i] = res

    def rotate(self):
        height = len(self.data)
        width = len(self.data[0])

        # Create a new empty list for rotated data
        rotated_data = [[0 for i in range(height)] for i in range(width)]

        for x in range(width):
            for y in range(height):
                rotated_data[x][y] = self.data[height - y - 1][x]

        # Update the data with the rotated data
        self.data = rotated_data

    def salt_n_pepper(self,amount=0.05):
        height = len(self.data)
        width = len(self.data[0])

        for x in range(width):
            for y in range(height):
                if random.random() < amount:
                    self.data[y][x] = 0
                elif random.random() < amount:
                    self.data[y][x] = 255

    def concat(self, other_img, direction='horizontal'):
        other_data = other_img.data
        height = min(len(self.data), len(other_data))
        width = min(len(self.data[0]), len(other_data[0]))

        if direction == 'horizontal':
            concatenated_data = [self.data[i][:width] + other_data[i][:width] for i in range(height)]
        else:  # direction == 'vertical'
            concatenated_data = self.data[:height] + other_data[:height]

        self.data = concatenated_data

    def segment(self, num_segments=4):
        height = len(self.data)
        len(self.data[0])
        segment_height = height // num_segments

        segments = []
        for i in range(num_segments):
            start_row = i * segment_height
            end_row = start_row + segment_height if i != num_segments - 1 else height
            segment = self.data[start_row:end_row]
            segments.append(segment)

        self.data = segments