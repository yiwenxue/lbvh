#include "camera.h"

#include "helper_math.h"

#define PI_OVER_TWO 1.5707963267948966192313216916397514420985
#define M_PI 3.1415926535897932384626433832795028841971

float mod(float x, float y)
{ // Does this account for -y ???
    return x - y * floorf(x / y);
}

float clamp2(float n, float low, float high)
{
    n = fminf(n, high);
    n = fmaxf(n, low);
    return n;
}

float radiansToDegrees(float radians)
{
    float degrees = radians * 180.0 / M_PI;
    return degrees;
}

float degreesToRadians(float degrees)
{
    float radians = degrees / 180.0 * M_PI;
    return radians;
}

// lookAt

float3 lookAt(float3 eye, float3 center, float3 up)
{
    float3 f = normalize(center - eye);
    float3 s = normalize(cross(f, up));
    float3 u = cross(s, f);
    return make_float3(s.x, u.x, -f.x);
}

Camera::Camera()
{
    resolution = make_int2(1280, 720);
    fov = make_float2(60, 60);
    centerPosition = make_float3(0, 0, 0);
    viewDirection = make_float3(0, 0, -1);
    
    yaw   = 0.0f;
    pitch = 0.3f;

    focalDistance  = 4.0;
    apertureRadius = 0.04;
    radius         = 4;
}

Camera::~Camera()
{
}

void Camera::strafe(float m)
{
    float3 strafeAxis = normalize(cross(viewDirection, make_float3(0, 1, 0)));
    // normalize
    strafeAxis = normalize(strafeAxis);
    centerPosition += strafeAxis * m;
}

void Camera::move(float m)
{
    centerPosition += viewDirection * m;
}

void Camera::changeYaw(float m)
{
    yaw += m;
    fixYaw();
}

void Camera::changePitch(float m)
{
    pitch += m;
    fixPitch();
}

void Camera::changeRadius(float m)
{
    radius += m;
    fixRadius();
}

void Camera::changeAltitude(float m)
{
    centerPosition.y += m;
}

void Camera::changeFocalDistance(float m)
{
    focalDistance += m;
    fixFocalDistance();
}

void Camera::changeAperture(float m)
{
    apertureRadius += m;
    fixAperture();
}

void Camera::setRes(int x, int y)
{
    resolution.x = x;
    resolution.y = y;
}

void Camera::setFovX(float x)
{
    fov.x = x;
    fov.y = radiansToDegrees(2 * atanf(tanf(degreesToRadians(x) * 0.5) * (resolution.y / resolution.x)));
}

void Camera::fixYaw()
{
    yaw = mod(yaw, 2 * M_PI);
}

void Camera::fixPitch()
{
    float padding = 0.05;
    pitch = clamp2(pitch, -PI_OVER_TWO + padding, PI_OVER_TWO - padding); // Limit the pitch.
}

void Camera::fixRadius()
{
    float minRadius = 0.2;
    float maxRadius = 100.0;
    radius = clamp2(radius, minRadius, maxRadius);
}

void Camera::fixAperture()
{
    float minApertureRadius = 0.0;
    float maxApertureRadius = 25.0;
    apertureRadius = clamp2(apertureRadius, minApertureRadius, maxApertureRadius);
}

void Camera::fixFocalDistance()
{
    float minFocalDist = 0.2;
    float maxFocalDist = 100.0;
    focalDistance = clamp2(focalDistance, minFocalDist, maxFocalDist);
}

void Camera::buildRenderCamera(camera_data &renderCamera)
{
    float xDirection = sin(yaw) * cos(pitch);
    float yDirection = sin(pitch);
    float zDirection = cos(yaw) * cos(pitch);
    float3 directionToCamera = make_float3(xDirection, yDirection, zDirection);
    viewDirection = directionToCamera * -1.0f;
    float3 eyePosition = centerPosition;

    renderCamera.pos = make_float4(eyePosition);
    renderCamera.view = make_float4(lookAt(eyePosition, centerPosition + viewDirection, make_float3(0, 1, 0)));
    renderCamera.fov = fov;
    renderCamera.up = make_float4(0, 1, 0, 0);
    renderCamera.aperture = apertureRadius;
    renderCamera.focalDist = focalDistance;
}
