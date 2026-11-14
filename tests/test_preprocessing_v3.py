"""Geometric and mask invariants for v3; no biometric performance claims."""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocessing_v3 import (
    ARCFACE_112, PATCH_NAMES, EYES, BROWS, MOUTH, NOSE,
    PreprocessingConfigV3, QualityRejection, PreparedFace,
    similarity_alignment, transform_points, largest_valid_rectangle, build_skin_regions,
)


def synthetic_landmarks():
    points=np.zeros((106,2),dtype=np.float32)
    angles=np.linspace(0,2*np.pi,33,endpoint=False)
    points[:33]=np.stack([112+90*np.cos(angles),135+84*np.sin(angles)],axis=1)
    points[0]=[112,218]
    for j in range(9,15):points[j]=[30+(j-9)*3,105+(j-9)*13]
    for j in range(25,31):points[j]=[194-(j-25)*3,105+(j-25)*13]
    def ellipse(indices,cx,cy,rx,ry):
        angle=np.linspace(0,2*np.pi,len(indices),endpoint=False)
        points[list(indices)]=np.stack([cx+rx*np.cos(angle),cy+ry*np.sin(angle)],axis=1)
    ellipse(EYES[0],75,103,14,5)
    ellipse(EYES[1],149,103,14,5)
    ellipse(BROWS[0],75,86,20,5)
    ellipse(BROWS[1],149,86,20,5)
    ellipse(MOUTH,112,184,24,9)
    ellipse(NOSE,112,143,12,20)
    return points


def test_similarity_recovers_known_scale_rotation_translation():
    target=ARCFACE_112*2
    angle=.31
    linear=1.4*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    source=target@linear.T+np.array([43,27])
    matrix,error=similarity_alignment(source,224)
    np.testing.assert_allclose(transform_points(source,matrix),target,atol=1e-8)
    assert error<1e-10
    assert np.linalg.det(matrix[:,:2])>0


def test_similarity_rejects_nonfinite_or_degenerate_landmarks():
    for points in [np.zeros((5,2)),np.full((5,2),np.nan),np.ones((6,2))]:
        with pytest.raises(QualityRejection):
            similarity_alignment(points)


def test_largest_rectangle_matches_exhaustive_small_mask_search():
    rng=np.random.default_rng(22)
    for _ in range(20):
        mask=(rng.random((5,6))>.3).astype(np.uint8)
        x1,y1,x2,y2=largest_valid_rectangle(mask)
        returned=(x2-x1)*(y2-y1)
        best=0
        for top in range(5):
            for bottom in range(top+1,6):
                for left in range(6):
                    for right in range(left+1,7):
                        if mask[top:bottom,left:right].all():
                            best=max(best,(bottom-top)*(right-left))
        assert returned==best
        if returned:assert mask[y1:y2,x1:x2].all()


def test_skin_mask_excludes_landmark_eye_brow_mouth_nose_regions():
    image=np.full((224,224,3),[110,150,190],dtype=np.uint8)
    points=synthetic_landmarks()
    skin,regions,polygons,quality=build_skin_regions(image,points,ARCFACE_112*2,PreprocessingConfigV3())
    for indices in (*EYES,*BROWS,MOUTH,NOSE):
        for x,y in np.rint(points[list(indices)]).astype(int):
            assert skin[y,x]==0
    assert all(np.all(mask<=skin) for mask in regions.values())
    assert quality["skin_mask_pixels"]>100
    assert "heuristic" in quality["mask_semantics"]


def test_warp_padding_never_becomes_valid_skin():
    image=np.full((224,224,3),[110,150,190],dtype=np.uint8)
    valid=np.ones((224,224),np.uint8)
    valid[:70]=0
    skin,regions,_,_=build_skin_regions(image,synthetic_landmarks(),ARCFACE_112*2,
                                      PreprocessingConfigV3(),valid)
    assert not skin[:70].any()
    assert all(not mask[:70].any() for mask in regions.values())


def test_cache_strict_crops_contain_only_valid_rectangle_and_no_pickle():
    rng=np.random.default_rng(30)
    patches={name:rng.integers(0,256,(64,64,3),dtype=np.uint8) for name in PATCH_NAMES}
    masks={name:np.zeros((64,64),dtype=np.uint8) for name in PATCH_NAMES}
    for mask in masks.values():mask[12:54,15:50]=1
    prepared=PreparedFace(np.zeros((224,224,3),np.uint8),synthetic_landmarks(),
                          (ARCFACE_112*2).astype(np.float32),np.ones((224,224),np.uint8),
                          patches,masks,np.array([[1.,0.,0.],[0.,1.,0.]]),
                          {"preprocessing_id":"synthetic-geometry-fixture"})
    arrays=prepared.numeric_arrays()
    for name in PATCH_NAMES:
        x1,y1,x2,y2=arrays["skin_rect_"+name]
        assert masks[name][y1:y2,x1:x2].all()
        assert (x1,y1,x2,y2)==(16,13,49,53)
        expected=cv2.resize(patches[name][y1:y2,x1:x2],(64,64),interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(expected,arrays["skin_only_"+name])
        assert np.all(arrays["masked_"+name][masks[name]==0]==128)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"sample.npz"
        prepared.save(path)
        with np.load(path,allow_pickle=False) as archive:
            assert all(not archive[key].dtype.hasobject for key in archive.files)
            assert json.loads(archive["metadata"].item())["preprocessing_id"]=="synthetic-geometry-fixture"


def test_empty_mask_has_explicit_empty_rect():
    assert largest_valid_rectangle(np.zeros((10,12),np.uint8))==(0,0,0,0)

