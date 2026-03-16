# Image Stitching

## Project Background
* 주제: **이미지 스티칭(Image Stitching)** 과정 직접 구현하기<br/>
* 제약 사항: 이미지 로딩과 저장, 기본적인 수학 연산을 제외한 전 과정의 라이브러리의 사용을 최소화<br/>
* 필수 구현: corner point 찾기, Point Matching, Homography 계산, Stitching<br/>
* 추가 정보: Homography 계산 과정에서는 Singular Value Decomposition(SVD)의 계산에 한해 라이브러리 사용이 허용<br/>
* 코드 작성 환경: Local computer, VSCode, Python


## Image preprocessing
image loading 및 noise 제거 과정을 수행. Gaussian Smoothing filter를 자용하여 중심 픽셀 값을 크게 가중. image 주변에 padding을 생성하여 가장자리 부근에서도 커널이 잘 지나갈 수 있도록 함. Gaussian kernel을 생성하고, 이미지 각 픽셀 주변 영역에서 커널을 통해 새로운 픽셀을 계산함. 

![preprocessing](https://github.com/coqls1229/cvproj/blob/main/src/preprocessing.png)

육안 상으로는 크게 달라진 것이 없도록 denoising을 진행.


## Corner detection
이미지의 기울기를 분석해서 x, y 양방향으로 기울기 변화가 큰 곳을 코너로 간주하는 **Harris Corner Detection**을 사용함. 이 과정에서는 soble filter를 사용해서 각 방향의 기울기를 구하고, Mean filter를 사용하여 이후 만들 M 행렬의 성분이 노이즈에 덜 민감하도록 만들어줌. 그런 다음 R값을 계산하고, threshold 값을 사용하여 corner의 위치들을 검출해냄.

![corner_detection](https://github.com/coqls1229/cvproj/blob/main/src/corner_detection.png)


## Point Matching
이전 단계에서 검출해낸 harris corner들을 토대로 NCC를 계산하여 matching을 진행함. 각 코너 포인트 주변에서 patch를 추출하여 NCC 유사성을 계산하고, 유사성 값이 Threshold 이상인 경우에 유효한 matching으로 이를 선택하도록 함.
하지만, testing 과정에서 point matching 부분의 코드를 2시간을 돌렸는데도 아무런 출력이 없었음. 그 원인을 이전 단계에서 검출해낸 코너 개수가 너무 많아서 연산량 증가로 인해 시간이 오래 소요된 것이라고 판단함. 따라서 Harris corner detection에서 기존의 threshold 값을 높이고, 상위 500개의 코너만을 사용하도록 코드를 수정하여 harris detection 단계를 다시 진행함. 

![point_matching](https://github.com/coqls1229/cvproj/blob/main/src/point_matching.png)

Ncc 를 사용하여 상위 200개 point들을 사용하여 point matching을 시도한 결과


## RANSAC
RANSAC을 사용해 이상치들을 걸러내는 과정 진행.

![ransac](https://github.com/coqls1229/cvproj/blob/main/src/ransac.png)


## Homography computation warping
RANSAC을 통해 선별해낸 point들을 사용하여 homography를 계산함. img2가 base 좌표계인 img1에 매핑된 결과를 확인하기 위해 warping을 수행하여 이가 제대로 계산되었는지 확인.

![warping](https://github.com/coqls1229/cvproj/blob/main/src/warping.png)

## Image Stitching
저장된 homography를 사용하여 파노라마 이미지를 생성.

![stitching](https://github.com/coqls1229/cvproj/blob/main/src/stitching.png)
