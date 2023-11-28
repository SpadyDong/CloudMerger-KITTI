#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;

static bool cmp(std::string a, std::string b)
{
    if (a < b)
    {
        return true;
    }
    return false;
}

static std::vector<std::string> pathList(std::string path, bool sort = false)
{
    std::vector<std::string> files;
    if (!boost::filesystem::exists(path))
    {
        std::cout << "dont exist" << std::endl;
        return files;
    }
    if (!boost::filesystem::is_directory(path))
    {
        std::cout << "not a dir" << std::endl;
        return files;
    }
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(path.c_str());

    while ((ptr = readdir(dir)) != NULL)
    {
        if (ptr->d_name[0] == '.')
            continue;
        files.push_back(path + "/" + ptr->d_name);
    }

    if (sort)
    {
        std::sort(files.begin(), files.end(), cmp);
    }

    return files;
}

std::vector<Eigen::Matrix<float, 4, 4>> readPose_KITTI(string path)
{
    std::vector<Eigen::Matrix<float, 4, 4>> poses;
    std::ifstream file(path);
    string temp_str;
    file >> temp_str;
    while (!file.eof())
    {
        Eigen::Matrix<float, 4, 4> pose_temp;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                pose_temp(i, j) = std::stof(temp_str);
                file >> temp_str;
            }
        }
        pose_temp.block<1, 4>(3, 0) = Eigen::Matrix<float, 1, 4>{0, 0, 0, 1};
        poses.push_back(pose_temp);
        if (temp_str.size() == 0)
        {
            break;
        }
    }
    file.close();

    return poses;
}

std::vector<Eigen::Matrix<float, 3, 4>> readCalib(string path)
{
    vector<Eigen::Matrix<float, 3, 4>> calib(5); // P0 P1 P2 P3 Tr

    std::ifstream file(path);
    string temp_str;
    for (int n = 0; n < 5; n++)
    {
        file >> temp_str;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                file >> temp_str;
                calib[n](i, j) = std::stof(temp_str);
            }
        }
    }
    file.close();

    return calib;
}

template <typename T>
float pointDepth(T p)
{
    float depth = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    return depth;
}

template <class T>
void depthCrop(T pc, float d)
{
    int i = 0;
    while (i < pc->size())
    {
        if (pointDepth(pc->points[i]) > d)
        {
            pc->erase(pc->begin() + i);
        }
        else
        {
            i++;
        }
    }
}

void readPointCloud(std::string fileName, pcl::PointCloud<pcl::PointXYZI>::Ptr pc)
{
    if (fileName.substr(fileName.length() - 4, 4) == ".bin" || fileName.substr(fileName.length() - 4, 4) == ".BIN")
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(fileName, *pc) != 0)
        {
            return;
        }
    }
    else if (fileName.substr(fileName.length() - 4, 4) == ".pcd" || fileName.substr(fileName.length() - 4, 4) == ".PCD")
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(fileName, *pc) != 0)
        {
            return;
        }
    }
    else if (fileName.substr(fileName.length() - 4, 4) == ".ply" || fileName.substr(fileName.length() - 4, 4) == ".PLY")
    {
        if (pcl::io::loadPLYFile<pcl::PointXYZI>(fileName, *pc) != 0)
        {
            return;
        }
    }

    return;
}

// 获取VELODYNE BIN文件中所包含的点数
int getBinSize(string path)
{
    int size = 0;
    FILE *fp = fopen(path.c_str(), "rb");
    if (fp)
    {
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fclose(fp);
    }
    size = size / (int)sizeof(float) / 4;
    return size;
}

// 读取KITTI VELODYNE点云
Eigen::MatrixXf readBin(string path, int size)
{
    Eigen::MatrixXf pc(size, 4);
    std::ifstream velodyne_bin(path, std::ios::binary);
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            float data;
            velodyne_bin.read((char *)&data, sizeof(float));
            pc(i, j) = data;
        }
    }
    velodyne_bin.close();
    return pc;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr readPointCloudKITTI(string fileName)
{
    Eigen::MatrixXf pc = readBin(fileName, getBinSize(fileName));
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (int i = 0; i < pc.rows(); i++)
    {
        pcl::PointXYZI temp;
        temp.x = pc(i, 0);
        temp.y = pc(i, 1);
        temp.z = pc(i, 2);
        cloud->push_back(temp);
    }
    return cloud;
}

Eigen::Matrix<float, 4, 4> inversePoseT(Eigen::Matrix<float, 4, 4> pose)
{
    pose.block<3,3>(0,0) = pose.block<3,3>(0,0).transpose();
    pose.block<3,1>(0,3) = pose.block<3,3>(0,0)*pose.block<3,1>(0,3);
    return pose;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr fastMerge(std::vector<std::string> lidar_files, std::vector<Eigen::Matrix<float, 4, 4>> poses, Eigen::Matrix<float, 4, 4> calib_lidar)
{
    assert(poses.size() == lidar_files.size());

    // 初始化，以第一帧的位姿为起始位姿
    Eigen::Matrix<float, 4, 4> init_pose_inv = inversePoseT(poses[0]);
    pcl::PointCloud<pcl::PointXYZI>::Ptr global_map = readPointCloudKITTI(lidar_files[0]);
    depthCrop(global_map, 50.0);
    pcl::transformPointCloud(*global_map, *global_map, calib_lidar);

    for (int i = 1; i < lidar_files.size(); i++)
    {
        // 采样，每一帧都处理，资源
        if(i%10!=0)
        {
            continue;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_temp = readPointCloudKITTI(lidar_files[i]);
        depthCrop(cloud_temp, 50.0);
        Eigen::Matrix<float, 4, 4> trans = init_pose_inv * poses[i] * calib_lidar;
        pcl::transformPointCloud(*cloud_temp, *cloud_temp, trans);
        *global_map += *cloud_temp;
    }

    pcl::VoxelGrid<pcl::PointXYZI> sor;
    float leaf_size = 0.2f;
    sor.setLeafSize(leaf_size, leaf_size, leaf_size);
    sor.setInputCloud(global_map);
    sor.filter(*global_map);

    return global_map;
}

int main()
{
    string velodyne_path = "/media/xwd/XWDSF306/KITTI/data_odometry_velodyne/dataset/sequences/00/velodyne";
    string pose_path = "/media/xwd/XWDSF306/KITTI/data_odometry_velodyne/dataset/poses/00.txt";
    string calib_path = "/media/xwd/XWDSF306/KITTI/data_odometry_calib/dataset/sequences/00/calib.txt";
    
    std::vector<std::string> lidar_files = pathList(velodyne_path, true);
    std::vector<Eigen::Matrix<float, 4, 4>> poses = readPose_KITTI(pose_path);
    std::vector<Eigen::Matrix<float, 3, 4>> calib = readCalib(calib_path);
    Eigen::Matrix<float, 4, 4> calib_lidar;
    calib_lidar.block<3, 4>(0, 0) = calib[4];
    calib_lidar.block<1, 4>(3, 0) = Eigen::Matrix<float, 1, 4>{0, 0, 0, 1};

    pcl::PointCloud<pcl::PointXYZI>::Ptr global_map = fastMerge(lidar_files, poses, calib_lidar);

    pcl::visualization::CloudViewer viewer("cloud view");
    viewer.showCloud(global_map);
    while (!viewer.wasStopped()){}

    pcl::io::savePCDFileASCII("kitti.pcd", *global_map);

    return 0;
}