class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.AMPK = 0

    def __repr__(self):
        return f"Cell({self.x}, {self.y})"

    def __str__(self):
        return f"Cell at ({self.x}, {self.y}) with AMPK level: {self.AMPK}"

    def increment_AMPK(self):
        self.AMPK += 1
        return self.AMPK
