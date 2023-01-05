//============================================================
// STUDENT NAME: Richard Willie
// NUS User ID.: E0550368
// COMMENTS TO GRADER:
//   - Developed on Manjaro
//   - Tested with GCC 12.2.0, GLUT 3.2.2, GLEW 2.2.0,
//     and OpenMP 4.5
//   - How to compile:
//       - cd <path to CMakeLists.txt>
//       - cmake -S ./ -B ./build
//       - cmake --build ./build
// ============================================================

#include "Camera.h"
#include "Color.h"
#include "Image.h"
#include "Light.h"
#include "Material.h"
#include "Plane.h"
#include "Ray.h"
#include "Raytrace.h"
#include "Scene.h"
#include "Sphere.h"
#include "Surface.h"
#include "Triangle.h"
#include "Util.h"
#include "Vector3d.h"
#include <string>

// Constants for Scene 1.
static constexpr int imageWidth1 = 640;
static constexpr int imageHeight1 = 480;
static constexpr int reflectLevels1 = 2; // 0 -- object does not reflect scene.
static constexpr int hasShadow1 = true;
static constexpr std::string_view outImageFile1 = "img_r2s.png";

// Constants for Scene 2.
static constexpr int imageWidth2 = 640;
static constexpr int imageHeight2 = 480;
static constexpr int reflectLevels2 = 2; // 0 -- object does not reflect scene.
static constexpr int hasShadow2 = true;
static constexpr std::string_view outImageFile2 = "img_scene2.png";

///////////////////////////////////////////////////////////////////////////
// Raytrace the whole image of the scene and write it to a file.
///////////////////////////////////////////////////////////////////////////

void RenderImage(const std::string &imageFilename, const Scene &scene,
                 int reflectLevels, bool hasShadow) {
  int imgWidth = scene.camera.getImageWidth();
  int imgHeight = scene.camera.getImageHeight();

  Image image(imgWidth, imgHeight); // To store the result of ray tracing.

  double startTime = Util::GetCurrRealTime();
  double startCPUTime = Util::GetCurrCPUTime();

// Generate image, rendering in parallel on Windows and Linux.
#ifndef __APPLE__
#pragma warning(push)
#pragma warning(disable : 6993)
#pragma omp parallel for
#endif
  for (int y = 0; y < imgHeight; y++) {
    double pixelPosY = y + 0.5;

    for (int x = 0; x < imgWidth; x++) {
      double pixelPosX = x + 0.5;
      Ray ray = scene.camera.getRay(pixelPosX, pixelPosY);
      Color pixelColor =
          Raytrace::TraceRay(ray, scene, reflectLevels, hasShadow);
      pixelColor.clamp();
      image.setPixel(x, y, pixelColor);
    }
  }
#ifndef __APPLE__
#pragma warning(pop)
#endif

  double cpuTimeElapsed = Util::GetCurrCPUTime() - startCPUTime;
  double realTimeElapsed = Util::GetCurrRealTime() - startTime;
  std::cout << "CPU time taken = " << cpuTimeElapsed << "sec" << std::endl;
  std::cout << "Real time taken = " << realTimeElapsed << "sec" << std::endl;

  // Write image to file.
  if (!image.writeToFile(imageFilename))
    return;
  else
    Util::ErrorExit("File: %s could not be written.\n", imageFilename.c_str());
}

// Forward declarations. These functions are defined later in the file.
void DefineScene1(Scene &scene, int imageWidth, int imageHeight);
void DefineScene2(Scene &scene, int imageWidth, int imageHeight);

int main() {
  /*
  // Define Scene 1.

  Scene scene1;
  DefineScene1(scene1, imageWidth1, imageHeight1);

  // Render Scene 1.

  std::cout << "Render Scene 1..." << std::endl;
  RenderImage(std::string(outImageFile1), scene1, reflectLevels1, hasShadow1);
  std::cout << "Scene 1 completed." << std::endl;

  // Delete Scene 1 surfaces.

  for (auto &surface : scene1.surfaces) {
    delete surface;
  }
  */

  // Define Scene 2.

  Scene scene2;
  DefineScene2(scene2, imageWidth2, imageHeight2);

  // Render Scene 2.

  std::cout << "Render Scene 2..." << std::endl;
  RenderImage(std::string(outImageFile2), scene2, reflectLevels2, hasShadow2);
  std::cout << "Scene 2 completed." << std::endl;

  // Delete Scene 2 surfaces.

  for (auto &surface : scene2.surfaces) {
    delete surface;
  }

  std::cout << "All done. Press Enter to exit." << std::endl;
  std::cin.get();
  return 0;
}

///////////////////////////////////////////////////////////////////////////
// Modelling of Scene 1.
///////////////////////////////////////////////////////////////////////////

void DefineScene1(Scene &scene, int imageWidth, int imageHeight) {
  scene.backgroundColor = Color(0.2f, 0.3f, 0.5f);

  scene.amLight.I_a = Color(1.0f, 1.0f, 1.0f) * 0.25f;

  // Define materials.

  // Light red.
  Material lightRed = Material();
  lightRed.k_d = Color(0.8f, 0.4f, 0.4f);
  lightRed.k_a = lightRed.k_d;
  lightRed.k_r = Color(0.8f, 0.8f, 0.8f) / 1.5f;
  lightRed.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  lightRed.n = 64.0f;

  // Light green.
  Material lightGreen = Material();
  lightGreen.k_d = Color(0.4f, 0.8f, 0.4f);
  lightGreen.k_a = lightGreen.k_d;
  lightGreen.k_r = Color(0.8f, 0.8f, 0.8f) / 1.5f;
  lightGreen.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  lightGreen.n = 64.0f;

  // Light blue.
  Material lightBlue = Material();
  lightBlue.k_d = Color(0.4f, 0.4f, 0.8f) * 0.9f;
  lightBlue.k_a = lightBlue.k_d;
  lightBlue.k_r = Color(0.8f, 0.8f, 0.8f) / 1.5f;
  lightBlue.k_rg = Color(0.8f, 0.8f, 0.8f) / 2.5f;
  lightBlue.n = 64.0f;

  // Yellow.
  Material yellow = Material();
  yellow.k_d = Color(0.6f, 0.6f, 0.2f);
  yellow.k_a = yellow.k_d;
  yellow.k_r = Color(0.8f, 0.8f, 0.8f) / 1.5f;
  yellow.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  yellow.n = 64.0f;

  // Gray.
  Material gray = Material();
  gray.k_d = Color(0.6f, 0.6f, 0.6f);
  gray.k_a = gray.k_d;
  gray.k_r = Color(0.6f, 0.6f, 0.6f);
  gray.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  gray.n = 128.0f;

  // Insert into scene materials vector.
  scene.materials = {lightRed, lightGreen, lightBlue, yellow, gray};

  // Define point light sources.

  scene.ptLights.resize(2);

  PointLightSource light0 = {Vector3d(100.0, 120.0, 10.0),
                             Color(1.0f, 1.0f, 1.0f) * 0.6f};
  PointLightSource light1 = {Vector3d(5.0, 80.0, 60.0),
                             Color(1.0f, 1.0f, 1.0f) * 0.6f};

  scene.ptLights = {light0, light1};

  // Define surface primitives.

  scene.surfaces.resize(15);

  auto horzPlane =
      new Plane(0.0, 1.0, 0.0, 0.0, scene.materials[2]); // Horizontal plane.
  auto leftVertPlane =
      new Plane(1.0, 0.0, 0.0, 0.0, scene.materials[4]); // Left vertical plane.
  auto rightVertPlane = new Plane(0.0, 0.0, 1.0, 0.0,
                                  scene.materials[4]); // Right vertical plane.
  auto bigSphere = new Sphere(Vector3d(40.0, 20.0, 42.0), 22.0,
                              scene.materials[0]); // Big sphere.
  auto smallSphere = new Sphere(Vector3d(75.0, 10.0, 40.0), 12.0,
                                scene.materials[1]); // Small sphere.

  // Cube +y face.
  auto cubePosYTri1 =
      new Triangle(Vector3d(50.0, 20.0, 90.0), Vector3d(50.0, 20.0, 70.0),
                   Vector3d(30.0, 20.0, 70.0), scene.materials[3]);
  auto cubePosYTri2 =
      new Triangle(Vector3d(50.0, 20.0, 90.0), Vector3d(30.0, 20.0, 70.0),
                   Vector3d(30.0, 20.0, 90.0), scene.materials[3]);

  // Cube +x face.
  auto cubePosXTri1 =
      new Triangle(Vector3d(50.0, 0.0, 70.0), Vector3d(50.0, 20.0, 70.0),
                   Vector3d(50.0, 20.0, 90.0), scene.materials[3]);
  auto cubePosXTri2 =
      new Triangle(Vector3d(50.0, 0.0, 70.0), Vector3d(50.0, 20.0, 90.0),
                   Vector3d(50.0, 0.0, 90.0), scene.materials[3]);

  // Cube -x face.
  auto cubeNegXTri1 =
      new Triangle(Vector3d(30.0, 0.0, 90.0), Vector3d(30.0, 20.0, 90.0),
                   Vector3d(30.0, 20.0, 70.0), scene.materials[3]);
  auto cubeNegXTri2 =
      new Triangle(Vector3d(30.0, 0.0, 90.0), Vector3d(30.0, 20.0, 70.0),
                   Vector3d(30.0, 0.0, 70.0), scene.materials[3]);

  // Cube +z face.
  auto cubePosZTri1 =
      new Triangle(Vector3d(50.0, 0.0, 90.0), Vector3d(50.0, 20.0, 90.0),
                   Vector3d(30.0, 20.0, 90.0), scene.materials[3]);
  auto cubePosZTri2 =
      new Triangle(Vector3d(50.0, 0.0, 90.0), Vector3d(30.0, 20.0, 90.0),
                   Vector3d(30.0, 0.0, 90.0), scene.materials[3]);

  // Cube -z face.
  auto cubeNegZTri1 =
      new Triangle(Vector3d(30.0, 0.0, 70.0), Vector3d(30.0, 20.0, 70.0),
                   Vector3d(50.0, 20.0, 70.0), scene.materials[3]);
  auto cubeNegZTri2 =
      new Triangle(Vector3d(30.0, 0.0, 70.0), Vector3d(50.0, 20.0, 70.0),
                   Vector3d(50.0, 0.0, 70.0), scene.materials[3]);

  scene.surfaces = {horzPlane,    leftVertPlane, rightVertPlane, bigSphere,
                    smallSphere,  cubePosXTri1,  cubePosXTri2,   cubePosYTri1,
                    cubePosYTri2, cubePosZTri1,  cubePosZTri2,   cubeNegXTri1,
                    cubeNegXTri2, cubeNegZTri1,  cubeNegZTri2};

  // Define camera.

  scene.camera = Camera(Vector3d(150.0, 120.0, 150.0),     // eye
                        Vector3d(45.0, 22.0, 55.0),        // lookAt
                        Vector3d(0.0, 1.0, 0.0),           // upVector
                        (-1.0 * imageWidth) / imageHeight, // left
                        (1.0 * imageWidth) / imageHeight,  // right
                        -1.0, 1.0, 3.0,                    // bottom, top, near
                        imageWidth, imageHeight); // image_width, image_height
}

///////////////////////////////////////////////////////////////////////////
// Modeling of Scene 2.
///////////////////////////////////////////////////////////////////////////

void DefineScene2(Scene &scene, int imageWidth, int imageHeight) {
  //***********************************************
  //*********** WRITE YOUR CODE HERE **************
  //***********************************************
  scene.backgroundColor = Color(0.2f, 0.3f, 0.5f);

  scene.amLight.I_a = Color(1.0f, 1.0f, 1.0f) * 0.25f;

  // Define materials.

  Material red = Material();
  red.k_d = Color(0.8f, 0.4f, 0.4f);
  red.k_a = red.k_d;
  red.k_r = Color(0.8f, 0.8f, 0.8f) / 1.5f;
  red.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  red.n = 64.0f;

  Material green = Material();
  green.k_d = Color(0.4f, 0.8f, 0.4f);
  green.k_a = green.k_d;
  green.k_r = Color(0.8f, 0.8f, 0.8f) / 1.5f;
  green.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  green.n = 64.0f;

  Material darkBlue = Material();
  darkBlue.k_d = Color(0.4f, 0.4f, 0.8f) * 0.9f;
  darkBlue.k_a = darkBlue.k_d;
  darkBlue.k_r = Color(0.8f, 0.8f, 0.8f) / 1.5f;
  darkBlue.k_rg = Color(0.8f, 0.8f, 0.8f) / 2.5f;
  darkBlue.n = 64.0f;

  Material yellow = Material();
  yellow.k_d = Color(0.6f, 0.6f, 0.2f);
  yellow.k_a = yellow.k_d;
  yellow.k_r = Color(0.8f, 0.8f, 0.8f) / 1.5f;
  yellow.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  yellow.n = 64.0f;

  Material gray = Material();
  gray.k_d = Color(0.6f, 0.6f, 0.6f);
  gray.k_a = gray.k_d;
  gray.k_r = Color(0.6f, 0.6f, 0.6f);
  gray.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  gray.n = 128.0f;

  Material pink = Material();
  pink.k_d = Color(0.8f, 0.4f, 0.6f);
  pink.k_a = pink.k_d;
  pink.k_r = Color(0.6f, 0.6f, 0.6f) / 1.5f;
  pink.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  pink.n = 64.0f;

  Material purple = Material();
  purple.k_d = Color(0.6f, 0.4f, 0.8f);
  purple.k_a = purple.k_d;
  purple.k_r = Color(0.6f, 0.6f, 0.6f) / 1.5f;
  purple.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  purple.n = 64.0f;

  Material orange = Material();
  orange.k_d = Color(0.9f, 0.6f, 0.4f);
  orange.k_a = orange.k_d;
  orange.k_r = Color(0.6f, 0.6f, 0.6f) / 1.5f;
  orange.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  orange.n = 64.0f;

  Material cyan = Material();
  cyan.k_d = Color(0.4f, 0.8f, 0.8f);
  cyan.k_a = cyan.k_d;
  cyan.k_r = Color(0.8f, 0.8f, 0.8f) / 1.5f;
  cyan.k_rg = Color(0.8f, 0.8f, 0.8f) / 2.5f;
  cyan.n = 64.0f;

  Material darkGray = Material();
  darkGray.k_d = Color(0.2f, 0.2f, 0.2f);
  darkGray.k_a = darkGray.k_d;
  darkGray.k_r = Color(0.6f, 0.6f, 0.6f);
  darkGray.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  darkGray.n = 128.0f;

  Material lightPink = Material();
  lightPink.k_d = Color(0.9f, 0.7f, 0.9f);
  lightPink.k_a = lightPink.k_d;
  lightPink.k_r = Color(0.6f, 0.6f, 0.6f) / 1.5f;
  lightPink.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  lightPink.n = 64.0f;

  Material blue = Material();
  blue.k_d = Color(0.4f, 0.6f, 0.8f);
  blue.k_a = blue.k_d;
  blue.k_r = Color(0.8f, 0.8f, 0.8f) / 1.5f;
  blue.k_rg = Color(0.8f, 0.8f, 0.8f) / 2.5f;
  blue.n = 64.0f;

  Material lightYellow = Material();
  lightYellow.k_d = Color(0.9f, 0.8f, 0.4f);
  lightYellow.k_a = lightYellow.k_d;
  lightYellow.k_r = Color(0.8f, 0.8f, 0.8f) / 1.5f;
  lightYellow.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  lightYellow.n = 64.0f;

  Material white = Material();
  white.k_d = Color(0.9f, 0.9f, 0.9f);
  white.k_a = white.k_d;
  white.k_r = Color(0.6f, 0.6f, 0.6f);
  white.k_rg = Color(0.8f, 0.8f, 0.8f) / 3.0f;
  white.n = 128.0f;

  // Insert into scene materials vector.
  scene.materials = {red,       green,  darkBlue,    yellow, gray,
                     pink,      purple, orange,      cyan,   darkGray,
                     lightPink, blue,   lightYellow, white};

  // Define point light sources.

  scene.ptLights.resize(2);

  PointLightSource light0 = {Vector3d(100.0, 120.0, 10.0),
                             Color(1.0f, 1.0f, 1.0f) * 0.6f};
  PointLightSource light1 = {Vector3d(5.0, 80.0, 60.0),
                             Color(1.0f, 1.0f, 1.0f) * 0.6f};

  scene.ptLights = {light0, light1};

  // Define surface primitives.

  auto horzPlane =
      new Plane(0.0, 1.0, 0.0, 0.0, scene.materials[9]); // Horizontal plane.
  auto leftVertPlane =
      new Plane(1.0, 0.0, 0.0, 0.0, scene.materials[9]); // Left vertical plane.
  auto rightVertPlane = new Plane(0.0, 0.0, 1.0, 0.0,
                                  scene.materials[9]); // Right vertical plane.

  // Cube +y face.
  auto cubePosYTri1 =
      new Triangle(Vector3d(40.0, 20.0, 110.0), Vector3d(40.0, 20.0, 90.0),
                   Vector3d(20.0, 20.0, 90.0), scene.materials[13]);
  auto cubePosYTri2 =
      new Triangle(Vector3d(40.0, 20.0, 110.0), Vector3d(20.0, 20.0, 90.0),
                   Vector3d(20.0, 20.0, 110.0), scene.materials[13]);

  // Cube +x face.
  auto cubePosXTri1 =
      new Triangle(Vector3d(40.0, 0.0, 90.0), Vector3d(40.0, 20.0, 90.0),
                   Vector3d(40.0, 20.0, 110.0), scene.materials[13]);
  auto cubePosXTri2 =
      new Triangle(Vector3d(40.0, 0.0, 90.0), Vector3d(40.0, 20.0, 110.0),
                   Vector3d(40.0, 0.0, 110.0), scene.materials[13]);

  // Cube -x face.
  auto cubeNegXTri1 =
      new Triangle(Vector3d(20.0, 0.0, 110.0), Vector3d(20.0, 20.0, 110.0),
                   Vector3d(20.0, 20.0, 90.0), scene.materials[13]);
  auto cubeNegXTri2 =
      new Triangle(Vector3d(20.0, 0.0, 110.0), Vector3d(20.0, 20.0, 90.0),
                   Vector3d(20.0, 0.0, 90.0), scene.materials[13]);

  // Cube +z face.
  auto cubePosZTri1 =
      new Triangle(Vector3d(40.0, 0.0, 110.0), Vector3d(40.0, 20.0, 110.0),
                   Vector3d(40.0, 20.0, 110.0), scene.materials[13]);
  auto cubePosZTri2 =
      new Triangle(Vector3d(40.0, 0.0, 110.0), Vector3d(20.0, 20.0, 110.0),
                   Vector3d(20.0, 0.0, 110.0), scene.materials[13]);

  // Cube -z face.
  auto cubeNegZTri1 =
      new Triangle(Vector3d(20.0, 0.0, 90.0), Vector3d(20.0, 20.0, 90.0),
                   Vector3d(40.0, 20.0, 90.0), scene.materials[13]);
  auto cubeNegZTri2 =
      new Triangle(Vector3d(20.0, 0.0, 90.0), Vector3d(40.0, 20.0, 90.0),
                   Vector3d(40.0, 0.0, 90.0), scene.materials[13]);

  auto sphere1 =
      new Sphere(Vector3d(30.0, 10.0, 30.0), 10.0, scene.materials[0]);

  auto sphere2 =
      new Sphere(Vector3d(85.0, 22.0, 30.0), 22.0, scene.materials[2]);

  auto sphere3 =
      new Sphere(Vector3d(50.0, 60.0, 40.0), 4.0, scene.materials[7]);

  auto sphere4 =
      new Sphere(Vector3d(56.0, 38.0, 50.0), 3.0, scene.materials[4]);

  auto sphere5 = new Sphere(Vector3d(60.0, 3.0, 82.0), 3.0, scene.materials[1]);

  auto sphere6 =
      new Sphere(Vector3d(34.0, 10.0, 52.0), 5.0, scene.materials[6]);

  auto sphere7 =
      new Sphere(Vector3d(78.0, 28.0, 80.0), 2.0, scene.materials[8]);

  auto sphere8 = new Sphere(Vector3d(40.0, 2.0, 72.0), 2.0, scene.materials[3]);

  auto sphere9 = new Sphere(Vector3d(58.0, 3.0, 40.0), 3.0, scene.materials[5]);

  auto sphere10 =
      new Sphere(Vector3d(10.0, 56.0, 78.0), 8.0, scene.materials[12]);

  auto sphere11 =
      new Sphere(Vector3d(3.0, 3.0, 60.0), 3.0, scene.materials[11]);

  auto sphere12 =
      new Sphere(Vector3d(84.0, 6.0, 74.0), 6.0, scene.materials[10]);

  scene.surfaces = {
      horzPlane,    leftVertPlane, rightVertPlane, cubePosXTri1, cubePosXTri2,
      cubePosYTri1, cubePosYTri2,  cubePosZTri1,   cubePosZTri2, cubeNegXTri1,
      cubeNegXTri2, cubeNegZTri1,  cubeNegZTri2,   sphere1,      sphere2,
      sphere3,      sphere4,       sphere5,        sphere6,      sphere7,
      sphere8,      sphere9,       sphere10,       sphere11,     sphere12};

  // Define camera.

  scene.camera = Camera(Vector3d(150.0, 120.0, 150.0),     // eye
                        Vector3d(45.0, 22.0, 55.0),        // lookAt
                        Vector3d(0.0, 1.0, 0.0),           // upVector
                        (-1.0 * imageWidth) / imageHeight, // left
                        (1.0 * imageWidth) / imageHeight,  // right
                        -1.0, 1.0, 3.0,                    // bottom, top, near
                        imageWidth, imageHeight); // image_width, image_height
}
