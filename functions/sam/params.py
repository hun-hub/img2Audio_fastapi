from params import RequestData

class SAM_RequestData(RequestData):
    sam_model: str = 'sam_vit_h_4b8939.pth'
    dino_model: str = 'GroundingDINO_SwinB (938MB)'
    image: str
    prompt: str
    threshold: float

