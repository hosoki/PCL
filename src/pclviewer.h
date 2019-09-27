#ifndef PCLVIEWER_H
#define PCLVIEWER_H

#include <iostream>
#include <vector>

// Qt
#include <QMainWindow>
#include <QImage>

#include <QFileDialog>
#include <QProgressDialog>
#include <QStandardPaths>
#include <QString>

// Point Cloud Library
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/face.hpp>
//#include <opencv2/core/core.hpp>


// Visualization Toolkit (VTK)
#include <vtkRenderWindow.h>



typedef pcl::PointXYZ PointXYZ;
typedef pcl::PointCloud<PointXYZ> CloudXYZ;
typedef pcl::PointXYZRGB PointXYZRGB;
typedef pcl::PointCloud<PointXYZRGB> CloudXYZRGB;
typedef pcl::PointXYZRGB PointXYZRGBN;
typedef pcl::PointCloud<PointXYZRGBN> CloudXYZRGBN;
typedef pcl::PolygonMesh PolygonMesh;
using namespace std;
using namespace cv;
using namespace cv::face;

namespace Ui
{
class PCLViewer;
}

class PCLViewer : public QMainWindow
{
    Q_OBJECT

public:
    explicit PCLViewer(QWidget *parent = 0);
    ~PCLViewer();

public slots:
    void
    load_point_cloud();

    void
    delete_point_cloud();

    void
    detect_face_landmark();

    void
    detect_symmetric_plane();

    void
    calc_asymmetry();

    void
    original_color();

    void
    paint_color();

    void
    RGBsliderReleased();

    void
    pSliderValueChanged(int value);

    void
    redSliderValueChanged(int value);

    void
    greenSliderValueChanged(int value);

    void
    blueSliderValueChanged(int value);

protected:
    boost::shared_ptr <pcl::visualization::PCLVisualizer> viewer;
    CloudXYZRGB::Ptr cloud;
    CloudXYZRGB::Ptr temp_cloud;
    CloudXYZRGB::Ptr cloud_filtered;
    CloudXYZRGB::Ptr cloud_asym;
    CloudXYZRGB::Ptr cloud_painted;

    cv::Mat img;

    bool color_mode = false;
    unsigned int red = 0;
    unsigned int green = 0;
    unsigned int blue = 0;
    unsigned int opacity = 1;

    CloudXYZRGB::Ptr cloud_in;
    CloudXYZRGB::Ptr cloud_target;
    CloudXYZRGB::Ptr cloud_source;
    CloudXYZRGB::Ptr cloud_source_trans;
    CloudXYZRGB::Ptr keypoints_t;
    CloudXYZRGB::Ptr keypoints_s;
    CloudXYZRGB::Ptr midpoints;
    pcl::PolygonMesh::Ptr plane;
    pcl::PolygonMesh::Ptr plane_ground_truth;
    pcl::PolygonMesh::Ptr plane_origin;

    pcl::IterativeClosestPoint<PointXYZRGB, PointXYZRGB>::Ptr icp;

    float midpoint[3] = { 0.0 };
    float plane_normal[3] = { 0.0 };
    float plane_delta[3] = { 0.0 };
    float plane_theta[3] = { 0.0 };
    float plane_ground_truth_delta[3] = { 0.0, 0.0, 0.0 };
    float plane_ground_truth_theta[3] = { 0.0, 0.0, 0.0 };
    float plane_error[3] = { 0.0 };

    std::string iteration_str;
    std::string score_str;
    std::string delta_x_str, theta_y_str, theta_z_str;
    std::string delta_x_gt_str, theta_y_gt_str, theta_z_gt_str;
    std::string delta_x_error_str, theta_y_error_str, theta_z_error_str;

    std::string iterations_cnt;
    std::string momentum_str;
    std::string ground_truth_str;
    std::string error_str;
    std::string unit_str;

private:
    Ui::PCLViewer *ui;

};

#endif // PCLVIEWER_H
