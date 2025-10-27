from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from app.schemas.payloads import SubmitRequest, SubmitResponse
from app.utils.image_ops import (
    b64_to_pil, pil_to_np_rgba, b64_to_mask_gray,
    resize_mask_nn, extract_contours_per_label,
    contours_to_svg, svg_to_b64,
    estimate_tilt_angle, rotate_image_and_mask, rotate_points
)

app = FastAPI(title="Face Contour API", version="1.2.0")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/api/v1/frontal/crop/submit", response_model=SubmitResponse)
def submit(req: SubmitRequest):
    # 1) decode
    try:
        pil_img = b64_to_pil(req.image)
        np_img_rgba = pil_to_np_rgba(pil_img)
        mask_gray = b64_to_mask_gray(req.segmentation_map)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image(s): {e}")

    H, W = np_img_rgba.shape[:2]
    mh, mw = mask_gray.shape[:2]

    # 2) ensure size match (resize mask if needed, preserves labels)
    if (W, H) != (mw, mh):
        mask_gray = resize_mask_nn(mask_gray, W, H)

    # 3) AUTO tilt correction (no schema changes)
    landmarks_xy = [(lm.x, lm.y) for lm in req.landmarks]
    angle = estimate_tilt_angle(landmarks_xy)
    M = None
    if angle is not None and abs(angle) > 1.0:
        # rotate by -angle to level the face
        np_img_rgba, mask_gray, M = rotate_image_and_mask(np_img_rgba, mask_gray, -angle)
        landmarks_xy = rotate_points(landmarks_xy, M)

    # 4) contours
    contours_map = extract_contours_per_label(mask_gray)

    # 5) JSON-safe contours
    mask_contours_serializable = {}
    for label, cnts in contours_map.items():
        mask_contours_serializable[str(label)] = [
            cnt.reshape(-1, 2).astype(float).tolist() for cnt in cnts
        ]

    # 6) SVG (upright)
    svg_text = contours_to_svg(W, H, contours_map, landmarks_xy)
    svg_b64 = svg_to_b64(svg_text)

    return JSONResponse(content={
        "svg": svg_b64,
        "mask_contours": mask_contours_serializable
    })
