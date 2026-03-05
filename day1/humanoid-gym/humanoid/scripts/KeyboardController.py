from pynput import keyboard

class KeyboardController:
    def __init__(self):
        self.left_stick_x = 0
        self.left_stick_y = 0
        self.right_stick_x = 0
        self.right_stick_y = 0
        self.key_map = {
            '8': (0, 1), '5': (0, -1), '4': (-1, 0), '6': (1, 0),
            'i': (0, 1), 'k': (0, -1), '7': (-1, 0), '9': (1, 0)
        }
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char in self.key_map:
                if key.char in '8456':
                    self.left_stick_x, self.left_stick_y = self.key_map[key.char]
                elif key.char in 'i7k9':
                    self.right_stick_x, self.right_stick_y = self.key_map[key.char]
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key.char in self.key_map:
                if key.char in '8456':
                    self.left_stick_x, self.left_stick_y = 0, 0
                elif key.char in 'i7k9':
                    self.right_stick_x, self.right_stick_y = 0, 0
        except AttributeError:
            pass

    def get_stick_values(self):
        return {
            "left_stick_x": self.left_stick_x,
            "left_stick_y": self.left_stick_y,
            "right_stick_x": self.right_stick_x,
            "right_stick_y": self.right_stick_y
        }