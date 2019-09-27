#include "pclviewer.h"
#include "../build/ui_pclviewer.h"

#define radians(deg) (((deg)/360)*2*M_PI)
#define degrees(rad) (((rad)/2/M_PI)*360)
bool wait_flag = false;
bool stop_flag = false;

float background_color[3] = { 1.0, 1.0, 1.0 };

float text_color[3] = { 0.0, 0.0, 0.0 };

float target_point_size = 1.0;
float target_color[3] = { 0.0, 0.0, 0.0 };
float target_opacity = 1.0;

float source_point_size = 1.0;
float source_color[3] = { 0.0, 1.0, 0.0 };
float source_opacity = 1.0;

float keypoints_t_point_size = 10.0;
float keypoints_t_color[3] = { 0.0, 0.0, 1.0 };
float keypoints_t_opacity = 1.0;

float keypoints_s_point_size = 10.0;
float keypoints_s_color[3] = { 1.0, 0.0, 0.0 };
float keypoints_s_opacity = 1.0;

float midpoints_point_size = 10.0;
float midpoints_color[3] = { 1.0, 0.0, 1.0 };
float midpoints_opacity = 1.0;

float line_width = 1.5;
float line_color[3] = { 0.0, 0.0, 0.0 };

float plane_color[3] = { 0.5, 0.0, 0.0 };
float plane_opacity = 0.1;

float plane_ground_truth_color[3] = { 0.0, 0.0, 0.5 };
float plane_ground_truth_opacity = 0.1;

float voxel_size = 1.5;

int viewport1(1);
int viewport2(2);
int viewport3(3);

void make_ground_truth_plane(float plane_ground_truth_delta[], float plane_ground_truth_theta[], pcl::PolygonMesh::Ptr plane);
void create_points(CloudXYZRGB::Ptr points, int n);
void update_keypoints(CloudXYZRGB::Ptr& cloud, CloudXYZRGB::Ptr keypoints);
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing);
void voxel_down_sample(CloudXYZRGB::Ptr& cloud, CloudXYZRGB::Ptr cloud_filtered, float voxel_size);
void transformPolygonMesh(pcl::PolygonMesh::Ptr plane, const Eigen::Matrix4d& transform_matrix);
void calculate_center_of_gravity(CloudXYZRGB::ConstPtr cloud, pcl::PolygonMesh::Ptr sphere, Eigen::Matrix4d& transform_matrix, float center_of_gravity[]);
void update_midpoints_plane(CloudXYZRGB::ConstPtr keypoints_t, CloudXYZRGB::ConstPtr keypoints_s, CloudXYZRGB::Ptr midpoints, float normal[], float delta[], float theta[],
                            pcl::PolygonMesh::Ptr plane, pcl::PolygonMesh::Ptr plane_origin);
void search_kNN(CloudXYZRGB::Ptr cloud, CloudXYZRGB::Ptr cloud_kNN, PointXYZRGB point, unsigned int k, unsigned int* k_idx);
void hsv_to_rgb(uint8_t* r, uint8_t* g, uint8_t* b, double asym);
void calculate_asymmetry(CloudXYZRGB::Ptr cloud, CloudXYZRGB::Ptr cloud_asym, pcl::PointCloud<pcl::Normal>::Ptr normals);

PCLViewer::PCLViewer (QWidget *parent) :
    QMainWindow (parent),
    ui (new Ui::PCLViewer)
{
    ui->setupUi (this);
    this->setWindowTitle ("PCL viewer");

    // Setup the cloud pointer
    cloud.reset (new CloudXYZRGB);

    // Fill the cloud with some points
    if (pcl::io::loadPCDFile("D:/data/CleftLipPointCloud/pcd_not_duplicated/025Reshaped.pcd", *cloud) < 0)
    {
        PCL_ERROR("Error loading target cloud\n");
        return;
    }

    unsigned int img_width(512), img_height(512);
    cv::Mat img = cv::Mat::zeros(img_width, img_height, CV_8UC3);
    QImage qimage(img.data, img.cols, img.rows, QImage::Format_RGB888);
    qimage = qimage.rgbSwapped();
    ui->qPixmap->setPixmap(QPixmap::fromImage(qimage));

    // Set up the QVTK window
    viewer.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    viewer->setBackgroundColor(1, 1, 1);
    ui->qvtkWidget->SetRenderWindow (viewer->getRenderWindow ());
    viewer->setupInteractor (ui->qvtkWidget->GetInteractor (), ui->qvtkWidget->GetRenderWindow ());
    ui->qvtkWidget->update ();

    // Connect "random" button and the function
    connect (ui->pushButton_load,  SIGNAL (clicked ()), this, SLOT (load_point_cloud()));
    connect (ui->pushButton_delete,  SIGNAL (clicked ()), this, SLOT (delete_point_cloud()));
    connect (ui->pushButton_landmark,  SIGNAL (clicked ()), this, SLOT (detect_face_landmark()));
    connect (ui->pushButton_basis,  SIGNAL (clicked ()), this, SLOT (detect_symmetric_plane()));
    connect (ui->pushButton_asym,  SIGNAL (clicked ()), this, SLOT (calc_asymmetry()));

    // connect radio button and the function
    connect(ui->radioButton_original, SIGNAL(clicked()), this, SLOT(original_color()));
    connect(ui->radioButton_colored, SIGNAL(clicked()), this, SLOT(paint_color()));

    // Connect R,G,B sliders and their functions
    connect (ui->horizontalSlider_R, SIGNAL (valueChanged (int)), this, SLOT (redSliderValueChanged (int)));
    connect (ui->horizontalSlider_G, SIGNAL (valueChanged (int)), this, SLOT (greenSliderValueChanged (int)));
    connect (ui->horizontalSlider_B, SIGNAL (valueChanged (int)), this, SLOT (blueSliderValueChanged (int)));
    connect (ui->horizontalSlider_R, SIGNAL (sliderReleased ()), this, SLOT (RGBsliderReleased ()));
    connect (ui->horizontalSlider_G, SIGNAL (sliderReleased ()), this, SLOT (RGBsliderReleased ()));
    connect (ui->horizontalSlider_B, SIGNAL (sliderReleased ()), this, SLOT (RGBsliderReleased ()));

    // Connect point size slider
    connect (ui->horizontalSlider_p, SIGNAL (valueChanged (int)), this, SLOT (pSliderValueChanged (int)));

    pSliderValueChanged (2);
    viewer->resetCamera ();
    ui->qvtkWidget->update ();
}

void
PCLViewer::load_point_cloud()
{
    //QString strFileName = QFileDialog::getOpenFileName(this, tr("select pcd file"), QStandardPaths::writableLocation(QStandardPaths::DesktopLocation));
    QString qload_file_name = QFileDialog::getOpenFileName(this, tr("select pcd file"), "D:/data/CleftLipPointCloud");
    std::string load_file_name = qload_file_name.toStdString();

    // Setup the cloud pointer
    cloud.reset(new CloudXYZRGB);
    cloud_painted.reset(new CloudXYZRGB);

    // Fill the cloud with some points
    if (pcl::io::loadPCDFile(load_file_name, *cloud) < 0)
    {
        PCL_ERROR("Error loading cloud\n");
        return;
    }
    *cloud_painted = *cloud;

    viewer->removePointCloud("cloud");
    viewer->addPointCloud(cloud, "cloud");
    //pSliderValueChanged(2);
    viewer->resetCamera();
    ui->qvtkWidget->update();
}

void
PCLViewer::delete_point_cloud()
{
    viewer->removePointCloud ("cloud");
    viewer->removePointCloud("cloud_painted");
    viewer->removePolygonMesh("plane");
    //pSliderValueChanged (2);
    viewer->resetCamera ();
    ui->qvtkWidget->update ();
}

void PCLViewer::detect_face_landmark()
{

    //    double sum_x(0.0), sum_y(0.0);
    //    for (int i = 0; i < cloud->points.size(); ++i)
    //    {
    //        sum_x += cloud->points[i].x;
    //        sum_y += cloud->points[i].y;
    //    }
    //    double gravity_x = sum_x / cloud->points.size();
    //    double gravity_y = sum_y / cloud->points.size();



    //    temp_cloud.reset (new CloudXYZRGB);
    //    *temp_cloud = *cloud;
    //    for (int i = 0; i < cloud->points.size(); ++i)
    //    {
    //        temp_cloud->points[i].x -= (gravity_x - 256);
    //        temp_cloud->points[i].y -= (gravity_y - 256);
    //        temp_cloud->points[i].y = 512 - temp_cloud->points[i].y;
    //    }

    //    unsigned int img_width(512), img_height(512);
    //    cv::Mat img = cv::Mat::zeros(img_width, img_height, CV_8UC3);

    //    for (int i = 0; i < cloud->points.size(); ++i)
    //    {
    //        img.at<cv::Vec3b>(temp_cloud->points[i].y, temp_cloud->points[i].x)[0] = cloud->points[i].b;
    //        img.at<cv::Vec3b>(temp_cloud->points[i].y, temp_cloud->points[i].x)[1] = cloud->points[i].g;
    //        img.at<cv::Vec3b>(temp_cloud->points[i].y, temp_cloud->points[i].x)[2] = cloud->points[i].r;
    //    }

    temp_cloud.reset (new CloudXYZRGB);
    *temp_cloud = *cloud;
    cloud_filtered.reset(new CloudXYZRGB);
    pcl::VoxelGrid<PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(voxel_size, voxel_size, voxel_size);
    sor.filter(*cloud_filtered);
    double right(-100.0), left(100.0), top(-100.0), bottom(100.0);

    // calcurate rect including point cloud
    for (int i = 0; i < cloud_filtered->points.size(); ++i)
    {
        if (right > cloud_filtered->points[i].x) right = cloud_filtered->points[i].x;
        if (left < cloud_filtered->points[i].x) left = cloud_filtered->points[i].x;
        if (top > cloud_filtered->points[i].y) top = cloud_filtered->points[i].y;
        if (bottom < cloud_filtered->points[i].y) bottom = cloud_filtered->points[i].y;
    }

    double shift_vector_x = (right + left) / 2.0;
    double shift_vector_y = (top + bottom) / 2.0;
    int margin(100);
    double reduction_rate(0.0);
    unsigned int img_width(512), img_height(512);
    cv::Mat img = cv::Mat::zeros(img_width, img_height, CV_8UC3);
    vector<vector<double>> z_info(img_height, vector<double>(img_width));
    for (int j = 0; j < img_height; ++j)
    {
        for (int i = 0; i < img_width; ++i)
        {
            z_info[j][i] = 0;
        }
    }

    // shift point cloud
    for (int i = 0; i < temp_cloud->points.size(); ++i)
    {
        temp_cloud->points[i].x -= shift_vector_x;
        temp_cloud->points[i].y -= shift_vector_y;
        temp_cloud->points[i].x = -temp_cloud->points[i].x;
    }

    if ((right - left) / (top - bottom) < img_width / img_height)
        reduction_rate = (img_height - 2 * margin) / (top - bottom);
    else
        reduction_rate = (img_width - 2 * margin) / (right - left);

    int facelandmark_2Dx(0), facelandmark_2Dy(0);
    for (int i = 0; i < cloud->points.size(); ++i)
    {
        facelandmark_2Dx = (int)(reduction_rate * temp_cloud->points[i].x + img_width / 2);
        facelandmark_2Dy = (int)(reduction_rate * temp_cloud->points[i].y + img_height / 2);

        img.at<cv::Vec3b>(facelandmark_2Dy, facelandmark_2Dx)[0] = cloud->points[i].b;
        img.at<cv::Vec3b>(facelandmark_2Dy, facelandmark_2Dx)[1] = cloud->points[i].g;
        img.at<cv::Vec3b>(facelandmark_2Dy, facelandmark_2Dx)[2] = cloud->points[i].r;

        z_info[facelandmark_2Dy][facelandmark_2Dx] = cloud->points[i].z;
    }


    Mat marked_img;
    marked_img = img;

    // Load Face Detector
    CascadeClassifier faceDetector("../data/haarcascade_frontalface_alt2.xml");

    // Create an instance of Facemark
    Ptr<Facemark> facemark = FacemarkLBF::create();

    // Load landmark detector
    facemark->loadModel("../data/lbfmodel.yaml");

    // Variable to store a video marked_img and its grayscale
    Mat gray;

    // Find face
    vector<Rect> faces;
    // Convert marked_img to grayscale because
    // faceDetector requires grayscale image.
    cvtColor(marked_img, gray, COLOR_BGR2GRAY);

    // Detect faces
    faceDetector.detectMultiScale(gray, faces);

    // Variable for landmarks.
    // Landmarks for one face is a vector of points
    // There can be more than one face in the image. Hence, we
    // use a vector of vector of points.
    vector< vector<Point2f> > landmarks;

    // Run landmark detector
    bool success = facemark->fit(marked_img,faces,landmarks);

    if(success)
    {
        // If successful, render the landmarks on the face
        for ( size_t i = 0; i < faces.size(); i++ )
        {
            rectangle(marked_img,faces[i],Scalar( 255, 0, 0 ));
        }
        for (unsigned long i=0;i<faces.size();i++)
        {
            for(unsigned long k=0;k<landmarks[i].size();k++)
            {
                if (k == 30 || k == 36 || k == 48 || k == 54)
                {
                    drawMarker(marked_img,landmarks[i][k],Scalar(255,0,0), MARKER_CROSS, 10, 1, LINE_8);
                }
                else
                {
                    drawMarker(marked_img,landmarks[i][k],Scalar(0,0,255), MARKER_CROSS, 10, 1, LINE_8);
                }
            }
        }
    }

    // 3D landmarks
    CloudXYZRGB::Ptr face_landmark(new CloudXYZRGB);
    face_landmark->width    = 1;
    face_landmark->height   = 1;
    face_landmark->is_dense = false;
    face_landmark->points.resize (face_landmark->width * face_landmark->height);
    face_landmark->points[0].x = -(landmarks[0][36].x - img_width / 2) / reduction_rate + shift_vector_x;
    face_landmark->points[0].y = (landmarks[0][36].y - img_height / 2) / reduction_rate + shift_vector_y;
    face_landmark->points[0].z = z_info[landmarks[0][36].y][landmarks[0][36].x];
    viewer->addPointCloud(face_landmark, "face_landmark");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "face_landmark");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "face_landmark");
    ui->qvtkWidget->update();

    QImage qimage(marked_img.data, marked_img.cols, marked_img.rows, QImage::Format_RGB888);
    qimage = qimage.rgbSwapped();
    ui->qPixmap->setPixmap(QPixmap::fromImage(qimage));

}

void PCLViewer:: detect_symmetric_plane()
{
    // Setup the cloud pointer
    //cloud_in.reset(new CloudXYZRGB);
    cloud_target.reset(new CloudXYZRGB);
    cloud_source.reset(new CloudXYZRGB);
    cloud_source_trans.reset(new CloudXYZRGB);
    keypoints_t.reset(new CloudXYZRGB);
    keypoints_s.reset(new CloudXYZRGB);
    midpoints.reset(new CloudXYZRGB);
    plane.reset(new pcl::PolygonMesh);
    plane_ground_truth.reset(new pcl::PolygonMesh);
    plane_origin.reset(new pcl::PolygonMesh);


    if (pcl::io::loadOBJFile("D:/data/CleftLipPointCloud/objects/plane01.obj", *plane_origin) < 0)
    {
        PCL_ERROR("Error loading sphere\n");
        //      return (-1);
    }

    *plane = *plane_origin;
    *plane_ground_truth = *plane_origin;

    // Make Ground Truth Plane
    make_ground_truth_plane(plane_ground_truth_delta, plane_ground_truth_theta, plane_ground_truth);

    // Voxel down sampling
    voxel_down_sample(cloud, cloud_target, voxel_size);

    // Mirroring
    Eigen::Matrix4d mirroring_matrix = Eigen::Matrix4d::Identity();
    mirroring_matrix(0, 0) = -1.0;
    pcl::transformPointCloud(*cloud_target, *cloud_source, mirroring_matrix);

    // Select keypoint within target point cloud
    create_points(keypoints_t, 1);
    create_points(keypoints_s, 1);
    create_points(midpoints, 1);

    // Initialize source by transform
    Eigen::Matrix4d transform_matrix = Eigen::Matrix4d::Identity();
    transform_matrix(0, 3) = 50;
    transform_matrix(1, 3) = 50;
    transform_matrix(2, 3) = 50;
    pcl::transformPointCloud(*cloud_source, *cloud_source, transform_matrix);
    transform_matrix = Eigen::Matrix4d::Identity();
    float theta = 30.0;
    transform_matrix(0, 0) = cos(theta);
    transform_matrix(0, 1) = -sin(theta);
    transform_matrix(1, 0) = sin(theta);
    transform_matrix(1, 1) = cos(theta);
    //pcl::transformPointCloud(*cloud_source, *cloud_source, transform_matrix);
    *cloud_source_trans = *cloud_source;
    update_keypoints(cloud_target, keypoints_t);
    update_keypoints(cloud_source_trans, keypoints_s);

    // The Iterative Closest Point algorithm
    icp.reset(new pcl::IterativeClosestPoint<PointXYZRGB, PointXYZRGB>);
    int iterations = 0;
    icp->setInputTarget(cloud_target);
    icp->setInputSource(cloud_source_trans);
    icp->setMaximumIterations(1);

    viewer->addPolygonMesh(*plane, "plane");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, plane_color[0], plane_color[1], plane_color[2], "plane");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, plane_opacity, "plane");

    // Main loop
    while (iterations++ < 50)
    {
        // registration
        icp->align(*cloud_source_trans);

        if (icp->hasConverged())
        {
            //pcl::transformPointCloud(*cloud_source, *cloud_source_trans, icp->getFinalTransformation());
            update_keypoints(cloud_source_trans, keypoints_s);
            update_midpoints_plane(keypoints_t, keypoints_s, midpoints, plane_normal, plane_delta, plane_theta, plane, plane_origin);

            //            iteration_str = std::to_string(iterations);
            //            score_str = std::to_string(icp->getFitnessScore());
            //            delta_x_str = std::to_string(plane_delta[0]);
            //            theta_y_str = std::to_string(degrees(plane_theta[1]));
            //            theta_z_str = std::to_string(degrees(plane_theta[2]));

            //            plane_error[0] = plane_delta[0] - plane_ground_truth_delta[0];
            //            plane_error[1] = degrees(plane_theta[1]) - plane_ground_truth_theta[1];
            //            plane_error[2] = degrees(plane_theta[2]) - plane_ground_truth_theta[2];

            //            delta_x_error_str = std::to_string(plane_error[0]);
            //            theta_y_error_str = std::to_string(plane_error[1]);
            //            theta_z_error_str = std::to_string(plane_error[2]);

            //            iterations_cnt = "ICP iterations = " + iteration_str + "\nRMSE(Root Mean Square Error) = " + score_str;
            //            momentum_str = "result:\ndelta_x = " + delta_x_str + "\ntheta_y = " + theta_y_str + "\ntheta_z = " + theta_z_str;
            //            error_str = "error:\ndelta_x = " + delta_x_error_str + "\ntheta_y = " + theta_y_error_str + "\ntheta_z = " + theta_z_error_str;

            //            viewer->updateText(iterations_cnt, 10, 100, 16, text_color[0], text_color[1], text_color[2], "iterations_cnt");
            //            viewer->updateText(momentum_str, 10, 850, 16, text_color[0], text_color[1], text_color[2], "momentum_str");
            //            viewer->updateText(error_str, 400, 850, 16, text_color[0], text_color[1], text_color[2], "error_str");
            //            viewer->updateText("Running", 10, 70, 16, 1.0, 0.0, 0.0, "start_stop");
            viewer->updatePointCloud(cloud_source_trans, "source_face");
            //            viewer->updatePointCloud(keypoints_s, "keypoints_s");
            //            viewer->updatePointCloud(midpoints, "midpoints");
            //            viewer->removeShape("line");
            //            viewer->addLine(keypoints_t->points[0], keypoints_s->points[0], "line");
            //            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width, "line");
            //            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, line_color[0], line_color[1], line_color[2], "line");
            viewer->updatePolygonMesh(*plane, "plane");
            ui->qvtkWidget->update ();
        }
        else
        {
            std::cout << "Not converged." << std::endl;
        }
    }
    //    viewer->spinOnce();
    ui->qvtkWidget->update ();
}

void
PCLViewer::calc_asymmetry()
{
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(cloud_filtered);
    n.setInputCloud(cloud_filtered);
    n.setSearchMethod(tree);
    n.setKSearch(20);
    n.compute(*normals);

    cloud_asym.reset(new CloudXYZRGB);
    calculate_asymmetry(cloud_filtered, cloud_asym, normals);

    viewer->removePointCloud("cloud");
    viewer->addPointCloud(cloud_asym, "cloud_asym");
    ui->qvtkWidget->update();
}

void
PCLViewer::original_color()
{
    color_mode = false;
    viewer->removePointCloud("cloud_painted");
    viewer->addPointCloud(cloud, "cloud");
    ui->qvtkWidget->update();
}

void
PCLViewer::paint_color()
{
    color_mode = true;
    viewer->removePointCloud("cloud");
    viewer->addPointCloud(cloud_painted, "cloud_painted");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, red / 255.0, green / 255.0, blue / 255.0, "cloud_painted");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, "cloud_paitned");
    ui->qvtkWidget->update();
}

void
PCLViewer::RGBsliderReleased ()
{
    // Set the new color
    if (color_mode)
    {
        for (size_t i = 0; i < cloud_painted->size (); i++)
        {
            cloud_painted->points[i].r = red;
            cloud_painted->points[i].g = green;
            cloud_painted->points[i].b = blue;
        }
        viewer->updatePointCloud (cloud_painted, "cloud_painted");
        ui->qvtkWidget->update ();
    }
}

void
PCLViewer::pSliderValueChanged (int value)
{
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, value, "cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, value, "cloud_painted");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, value, "cloud_asym");
    ui->qvtkWidget->update ();

}

void
PCLViewer::redSliderValueChanged (int value)
{
    red = value;
    if (color_mode)
    {
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, red / 255.0, green / 255.0, blue / 255.0, "cloud_painted");
        ui->qvtkWidget->update();
    }
}

void
PCLViewer::greenSliderValueChanged (int value)
{
    green = value;
    if (color_mode)
    {
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, red / 255.0, green / 255.0, blue / 255.0, "cloud_painted");
        ui->qvtkWidget->update();
    }
}

void
PCLViewer::blueSliderValueChanged (int value)
{
    blue = value;
    if (color_mode)
    {
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, red / 255.0, green / 255.0, blue / 255.0, "cloud_painted");
        ui->qvtkWidget->update();
    }
}

PCLViewer::~PCLViewer ()
{
    delete ui;
}

void transformPolygonMesh(pcl::PolygonMesh::Ptr plane, const Eigen::Matrix4d& transform_matrix)
{
    pcl::PointCloud<pcl::PointXYZ> temp;
    pcl::fromPCLPointCloud2(plane->cloud, temp);
    pcl::transformPointCloud(temp, temp, transform_matrix);
    pcl::toPCLPointCloud2(temp, plane->cloud);
}

void make_ground_truth_plane(float plane_ground_truth_delta[], float plane_ground_truth_theta[], pcl::PolygonMesh::Ptr plane)
{
    Eigen::Matrix4d parallel_mat = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d rotation_y_mat = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d rotation_z_mat = Eigen::Matrix4d::Identity();

    parallel_mat(0, 3) = plane_ground_truth_delta[0];
    parallel_mat(1, 3) = plane_ground_truth_delta[1];
    parallel_mat(2, 3) = plane_ground_truth_delta[2];

    rotation_y_mat(0, 0) = cos(radians(plane_ground_truth_theta[1]));
    rotation_y_mat(0, 2) = sin(radians(plane_ground_truth_theta[1]));
    rotation_y_mat(2, 0) = -sin(radians(plane_ground_truth_theta[1]));
    rotation_y_mat(2, 2) = cos(radians(plane_ground_truth_theta[1]));

    rotation_z_mat(0, 0) = cos(radians(plane_ground_truth_theta[2]));
    rotation_z_mat(0, 1) = -sin(radians(plane_ground_truth_theta[2]));
    rotation_z_mat(1, 0) = sin(radians(plane_ground_truth_theta[2]));
    rotation_z_mat(1, 1) = cos(radians(plane_ground_truth_theta[2]));


    transformPolygonMesh(plane, rotation_y_mat);
    transformPolygonMesh(plane, rotation_z_mat);
    transformPolygonMesh(plane, parallel_mat);
}

void create_points(CloudXYZRGB::Ptr points, int n)
{
    points->width = n;
    points->height = 1;
    points->is_dense = false;
    points->points.resize(points->width * points->height);

    for (size_t i = 0; i < points->points.size(); i++)
    {
        points->points[i].x = 0.0;
        points->points[i].y = 0.0;
        points->points[i].z = 0.0;
    }
}

void update_keypoints(CloudXYZRGB::Ptr& cloud, CloudXYZRGB::Ptr keypoints)
{
    for (size_t i = 0; i < keypoints->points.size(); i++)
    {
        keypoints->points[i].x = cloud->points[4000].x;
        keypoints->points[i].y = cloud->points[4000].y;
        keypoints->points[i].z = cloud->points[4000].z;
    }
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* nothing)
{
    if (event.getKeySym() == "space" && event.keyDown())
        wait_flag = !wait_flag;
}

void voxel_down_sample(CloudXYZRGB::Ptr& cloud, CloudXYZRGB::Ptr cloud_filtered, float voxel_size)
{
    pcl::VoxelGrid<PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(voxel_size, voxel_size, voxel_size);
    sor.filter(*cloud_filtered);
}

void calculate_center_of_gravity(CloudXYZRGB::ConstPtr cloud, pcl::PolygonMesh::Ptr sphere, Eigen::Matrix4d& transform_matrix, float center_of_gravity[])
{
    float position[3] = { 0.0 };
    transform_matrix(0, 3) = -transform_matrix(0, 3);
    transform_matrix(1, 3) = -transform_matrix(1, 3);
    transform_matrix(2, 3) = -transform_matrix(2, 3);
    transformPolygonMesh(sphere, transform_matrix);

    for (size_t i = 0; i < cloud->points.size(); i++)
    {
        position[0] += cloud->points[i].x;
        position[1] += cloud->points[i].y;
        position[2] += cloud->points[i].z;
    }

    center_of_gravity[0] = position[0] / float(cloud->points.size());
    center_of_gravity[1] = position[1] / float(cloud->points.size());
    center_of_gravity[2] = position[2] / float(cloud->points.size());

    transform_matrix(0, 3) = center_of_gravity[0];
    transform_matrix(1, 3) = center_of_gravity[1];
    transform_matrix(2, 3) = center_of_gravity[2];

    transformPolygonMesh(sphere, transform_matrix);
}

void update_midpoints_plane(CloudXYZRGB::ConstPtr keypoints_t, CloudXYZRGB::ConstPtr keypoints_s, CloudXYZRGB::Ptr midpoints, float normal[], float delta[], float theta[],
                            pcl::PolygonMesh::Ptr plane, pcl::PolygonMesh::Ptr plane_origin)
{
    float norm;
    Eigen::Matrix4d parallel_mat = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d rotation_y_mat = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d rotation_z_mat = Eigen::Matrix4d::Identity();


    midpoints->points[0].x = (keypoints_t->points[0].x + keypoints_s->points[0].x) / 2.0;
    midpoints->points[0].y = (keypoints_t->points[0].y + keypoints_s->points[0].y) / 2.0;
    midpoints->points[0].z = (keypoints_t->points[0].z + keypoints_s->points[0].z) / 2.0;

    normal[0] = (keypoints_t->points[0].x - keypoints_s->points[0].x);
    normal[1] = (keypoints_t->points[0].y - keypoints_s->points[0].y);
    normal[2] = (keypoints_t->points[0].z - keypoints_s->points[0].z);

    norm = sqrt(pow(normal[0], 2.0) + pow(normal[1], 2.0) + pow(normal[2], 2.0));
    normal[0] = normal[0] / norm;
    normal[1] = normal[1] / norm;
    normal[2] = normal[2] / norm;

    delta[0] = midpoints->points[0].x;
    delta[1] = midpoints->points[0].y;
    delta[2] = midpoints->points[0].z;

    theta[0] = -atan(normal[2] / normal[1]);
    theta[1] = -atan(normal[2] / normal[0]);
    theta[2] = atan(normal[1] / normal[0]);

    parallel_mat(0, 3) = delta[0];
    /*parallel_mat(1, 3) = delta[1];
    parallel_mat(2, 3) = delta[2];*/

    rotation_y_mat(0, 0) = cos(theta[1]);
    rotation_y_mat(0, 2) = sin(theta[1]);
    rotation_y_mat(2, 0) = -sin(theta[1]);
    rotation_y_mat(2, 2) = cos(theta[1]);

    rotation_z_mat(0, 0) = cos(theta[2]);
    rotation_z_mat(0, 1) = -sin(theta[2]);
    rotation_z_mat(1, 0) = sin(theta[2]);
    rotation_z_mat(1, 1) = cos(theta[2]);

    *plane = *plane_origin;

    transformPolygonMesh(plane, rotation_y_mat);
    transformPolygonMesh(plane, rotation_z_mat);
    transformPolygonMesh(plane, parallel_mat);
}

void search_kNN(CloudXYZRGB::Ptr cloud, CloudXYZRGB::Ptr cloud_kNN, PointXYZRGB point, unsigned int k, unsigned int* k_idx)
{
    // KD�؂����Ă����B�ߖT�_�T���Ƃ��������Ȃ�B
    pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZRGB>);
    tree->setInputCloud(cloud);

    //�ߖT�_�T���Ɏg���p�����[�^�ƌ��ʂ�����ϐ�
    double radius = 100;  //���ar
    std::vector<int> k_indices;  //�͈͓�̓_�̃C���f�b�N�X������
    std::vector<float> k_sqr_distances;  //�͈͓�̓_�̋���������
    unsigned int max_nn = 30;


    //���ar�ȓ�ɂ���_��T��
    tree->radiusSearch(point, radius, k_indices, k_sqr_distances, max_nn);

    if (k_indices.size() == 0) return;

    cloud_kNN->width = k;
    cloud_kNN->height = 1;
    cloud_kNN->is_dense = false;
    cloud_kNN->points.resize(cloud_kNN->width * cloud_kNN->height);

    *k_idx = k_indices[0];

    for (size_t i = 0; i < cloud_kNN->points.size(); ++i)
    {
        cloud_kNN->points[i].x = cloud->points[k_indices[i]].x;
        cloud_kNN->points[i].y = cloud->points[k_indices[i]].y;
        cloud_kNN->points[i].z = cloud->points[k_indices[i]].z;

        //std::cout << cloud->points[k_indices[i]].x << " " << cloud->points[k_indices[i]].y << " " << cloud->points[k_indices[i]].z << endl;
        //std::cout << cloud_kNN->points[i].x << " " << cloud_kNN->points[i].y << " " << cloud_kNN->points[i].z << endl;

        //std::cout << k_indices[i] << std::endl;
    }

}

void hsv_to_rgb(uint8_t* r, uint8_t* g, uint8_t* b, double asym)
{
    double asym_max = 20;
    double asym_min = 0;
    double asym_a = 240 / (asym_min - asym_max);
    double asym_b = -asym_max * asym_a;

    double h;
    //if (asym > asym_max) h = 0;
    if (asym > asym_max) h = 240;
    else if (asym < asym_min) h = 240;
    else h = asym_a * asym + asym_b;

    double s = 255;
    double v = 255;
    double max = v;
    double min = max - ((s / 255.0) * max);

    if (h >= 360) h = h - 360;

    if (0 <= h && h < 60)
    {
        *r = max;
        *g = (h / 60) * (max - min) + min;
        *b = min;
    }
    else if (60 <= h && h < 120)
    {
        *r = ((120 - h) / 60) * (max - min) + min;
        *g = max;
        *b = min;
    }
    else if (120 <= h && h < 180)
    {
        *r = min;
        *g = max;
        *b = ((h - 120) / 60) * (max - min) + min;
    }
    else if (180 < h && h < 240)
    {
        *r = min;
        *g = ((240 - h) / 60) * (max - min) + min;
        *b = max;
    }
    else if (240 <= h && h < 300)
    {
        *r = ((h -240) / 60) * (max - min) + min;
        *g = min;
        *b = max;
    }
    else
    {
        *r = max;
        *g = min;
        *b = ((360 - h) / 60) * (max - min) + min;
    }

}

void calculate_asymmetry(CloudXYZRGB::Ptr cloud, CloudXYZRGB::Ptr cloud_asym, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
    CloudXYZRGB::Ptr cloud_kNN(new CloudXYZRGB);
    PointXYZRGB p;
    double z;
    double asymmetry;
    uint8_t r, g, b;
    uint32_t rgb;

    cloud_asym->width = cloud->width;
    cloud_asym->height = 1;
    cloud_asym->is_dense = false;
    cloud_asym->points.resize(cloud_asym->width * cloud_asym->height);


    int num_iteration = cloud->points.size();
    QProgressDialog progress("Task in progress...", "Cancel", 0, num_iteration);
    progress.setWindowModality(Qt::WindowModal);


    unsigned int k_idx;
    double cos_normal;
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        progress.setValue(i);
        if (progress.wasCanceled())
            break;

        p.x = -cloud->points[i].x;
        p.y = cloud->points[i].y;
        p.z = cloud->points[i].z;
        search_kNN(cloud, cloud_kNN, p, 1, &k_idx);

        /*make_triangle(cloud_kNN->points[0], cloud_kNN->points[1], cloud_kNN->points[2], p.x, p.y, &z);
        p.z = z;*/

        // compare normal with inverse one
        PointXYZRGB no;
        PointXYZRGB nr;

        no.x = -normals->points[i].normal_x;
        no.y = normals->points[i].normal_y;
        no.z = normals->points[i].normal_z;

        nr.x = normals->points[k_idx].normal_x;
        nr.y = normals->points[k_idx].normal_y;
        nr.z = normals->points[k_idx].normal_z;

        cos_normal = (no.x * nr.x + no.y * nr.y + no.z * nr.z) / (sqrt(no.x * no.x + no.y * no.y + no.z * no.z) * sqrt(nr.x * nr.x + nr.y * nr.y + nr.z * nr.z));
        cos_normal = degrees(acos(cos_normal));

        std::cout << std::setw(6) << i << "/" << cloud->points.size() <<
                     " : original point : " << std::showpoint << std::setw(8) << std::internal << cloud->points[i].x <<
                     " reverse point : " << std::showpoint << std::setw(8) << std::internal << cloud->points[k_idx].x <<
                     " cos_normal : " << std::showpoint << std::setw(8) << std::internal << cos_normal << std::endl;

        asymmetry = cos_normal;
        //asymmetry = fabs(cloud->points[i].z - p.z);

        cloud_asym->points[i].x = cloud->points[i].x;
        cloud_asym->points[i].y = cloud->points[i].y;
        cloud_asym->points[i].z = cloud->points[i].z;

        hsv_to_rgb(&r, &g, &b, asymmetry);

        uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        cloud_asym->points[i].rgb = *reinterpret_cast<float*>(&rgb);

        /*std::cout << std::setw(6) << i << "/" << cloud->points.size() <<
            " : original point : " << std::showpoint << std::setw(8) << std::internal << cloud->points[i].z <<
            " reverse point : " << std::showpoint << std::setw(8) << std::internal << p.z <<
            " accuracy : " << std::showpoint << std::setw(8) << std::internal << asymmetry << std::endl;*/

        /*std::cout << std::setw(6) << i << "/" << cloud->points.size() <<
            " : original point : " << std::fixed << std::setprecision(4) << std::setw(10) << cloud->points[i].z <<
            " reverse point : " << std::fixed << std::setprecision(4) << std::setw(10) << p.z <<
            " accuracy : " << std::fixed << std::setprecision(4) << std::setw(10) << asymmetry << std::endl;*/
    }
    progress.setValue(num_iteration);
    //std::cout << "calculate symmetry finished" <<  std::endl;
}
