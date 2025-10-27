from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Landmark(BaseModel):
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)

class SubmitRequest(BaseModel):
    image: str  # base64-encoded original image (any mode)
    landmarks: List[Landmark]  # any count
    segmentation_map: str  # base64-encoded label mask (binary or multi-label)
    upright_svg: Optional[bool] = False  # if True, rotate to make eyes horizontal

class SubmitResponse(BaseModel):
    svg: str  # base64-encoded SVG content
    mask_contours: Dict[str, Any]  # label -> list of contours (each: [ [x,y], ... ])
