# define our data structure
class BoundingBox():
    def __init__(self, xmin, ymin, xmax, ymax):
        """
            Args:
                xmin (float): minimal value of x axis of bounding box, either (0 ~ 1) or (0 ~ img_w)
                ymin (float): minimal value of y axis of bounding box, either (0 ~ 1) or (0 ~ img_h)
                xmax (float): maximal value of x axis of bounding box, either (0 ~ 1) or (0 ~ img_w)
                ymax (float): maximal value of y axis of bounding box, either (0 ~ 1) or (0 ~ img_h)
        """
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.xmax = float(xmax)
        self.ymax = float(ymax)
        assert(self.xmax>=self.xmin), "xmax must equal or be greater than xmin"
        assert(self.ymax>=self.ymin), "ymax must equal or be greater than ymin"
        self._area()

    def _area(self):
        """calculate the area of bounding box
        """
        self.area = (self.xmax-self.xmin)*(self.ymax-self.ymin)
        assert(self.area>=0), "Area must be greater than 0"

class GroundTruthBoundingBox(BoundingBox):
    def __init__(self, image_id, class_id, xmin, ymin, xmax, ymax):
        """
            Args:
                image_id: image ID (str)
                class_id: class ID (int)
        """
        super().__init__(xmin, ymin, xmax, ymax)
        self.image_id = str(image_id)
        self.class_id = int(float(class_id))

class PredictBoundingBox(BoundingBox):
    def __init__(self, image_id, class_id, conf, xmin, ymin, xmax, ymax):
        """
            Args:
                image_id: image ID (str)
                class_id: class ID (int)
                conf: the confidence score of bounding box belongs to the class (float 0 ~ 1)
        """
        super().__init__(xmin, ymin, xmax, ymax)
        self.image_id = str(image_id)
        self.class_id = int(float(class_id))
        self.conf = float(conf)


