#pragma once

#include <cuda_runtime.h>

/**
 * @brief A simple camera class
 *
 */
struct camera_data
{
  float4 pos;      // Camera position
  float4 view;     // View direction
  float4 up;       // Up direction
  float2 fov;      // Field of view
  float aperture;  // Aperture
  float focalDist; // Focal distance
};

class Camera
{
public:
  Camera();
  virtual ~Camera();

  void buildRenderCamera(camera_data &renderCamera);

  void updateCameraUBO();

  void changeYaw(float m);
  void changePitch(float m);
  void changeRadius(float m);
  void changeAltitude(float m);
  void changeFocalDistance(float m);

  void changeAperture(float m);

  void setRes(int x, int y);
  void setFovX(float x);

  void move(float m);
  void strafe(float m);
  void rotate(){}; // TODO yiwenxue : not implemented

  int2 resolution;
  float2 fov;

  float3 centerPosition;
  float3 viewDirection;
  float yaw{0};
  float pitch{0};
  float radius{0};
  float apertureRadius{0};
  float focalDistance{0};


private:
  void fixYaw();
  void fixPitch();
  void fixAperture();
  void fixRadius();
  void fixFocalDistance();

};
