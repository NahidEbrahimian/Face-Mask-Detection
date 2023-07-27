class Face:
    """
    Base class for detected face objects
      
    Attributes:
        bbox (np.array): np array with shape (4,) and in [top left x, top left y, bottom right x, bottom right y] format (Essential)
        score (float): confidence score of detected face
        kps (np.array): np array with shape (5, 2) for 5 keypoints (using for face alignment)
        embedding (np.array): np array with shape (512,) for reperesentation vector
        identity (string): name of identified person
        cropped_face (np.aray): np array with shape (h, w, 3), cropped face bgr image
    """
    
    def __init__(self, bbox, score, kps, cropped_face=None, embedding=None, identity="unknown"):
        """
        The constructor for Face class.
  
        Parameters:
            bbox (np.array): np array with shape (4,) and in [top left x, top left y, bottom right x, bottom right y] format (Essential)
            score (float): confidence score of detected face
            kps (np.array): np array with shape (5, 2) for 5 keypoints (using for face alignment)
            cropped_face (np.aray): np array with shape (h, w, 3), cropped face bgr image
            embedding (np.array): np array with shape (512,) for reperesentation vector
            identity (string): name of identified person
            similarity (float): similaity to the source face, between -1 and +1
        """
        self.bbox = bbox
        self.score = score
        self.kps = kps
        self.cropped_face = cropped_face
        self.embedding = embedding
        self.identity = identity
        self.similarity = -1.
  
    def to_tlwh(self):
        """
        Helper function for converting [top left x, top left y, bottom right x, bottom right y] (default) format
        to [top left x, top left y, width, height] format
  
        Returns:
            tlwh_bbox (np.array): bounding box in [top left x, top left y, width, height] format 
        """
        tlwh_bbox = self.bbox.copy()
        tlwh_bbox[2:] = tlwh_bbox[2:] - tlwh_bbox[:2]
        
        return tlwh_bbox

    def to_xywh(self):
        """
        Helper function for converting [top left x, top left y, bottom right x, bottom right y] (default) format
        to [center x, center y, width, height] format
  
        Returns:
            xywh_bbox (np.array): bounding box in [center x, center y, width, height] format 
        """
        xywh_bbox = self.to_tlwh()
        xywh_bbox[:2] = xywh_bbox[:2] + xywh_bbox[2:]/2
        
        return xywh_bbox

    def to_xyah(self):
        """
        Helper function for converting [top left x, top left y, bottom right x, bottom right y] (default) format
        to [center x, center y, aspec ratio, height] format
  
        Returns:
            xyah_bbox (np.array): bounding box in [center x, center y, aspect_ratio, height] format, where the aspect_ratio is width / height
        """
        xyah_bbox = self.to_xywh()
        xyah_bbox[2] = xyah_bbox[2] / xyah_bbox[3]
        
        return xyah_bbox