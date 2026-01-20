
class EmotionIntensityRegulator:
    def __init__(self, mythrunner):
        self.mythrunner  =  mythrunner

    def get_max_intensity(self):
        if self.mythrunner.check_state("ego_phase_active"):
            return 6
        return 4

    def regulate(self, intensity):
        return min(intensity, self.get_max_intensity())
