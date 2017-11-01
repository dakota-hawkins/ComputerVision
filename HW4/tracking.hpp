// Standard Library Imports
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <math.h>
#include <map>
#include <set>
#include <algorithm> // remove

// OpenCV
#include <opencv2/opencv.hpp>

// BOOST
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string_regex.hpp>
#include <boost/regex.hpp>

// Hungarian
#include "./hungarian-algorithm-cpp-master/Hungarian.h"

//Previous Work
#include "../HW3/BinaryImageAnalysis/banalysis.cpp"


/*!
analyze bats
*/
void analyze_bats();

/*!
Convert a .csv file to an image file.

Used to convert bat segmentations into images.

@param csv_file: path to comma separated file representing image segmentations.

@return 8-bit image of mapped csv values. 
*/
cv::Mat csv_to_img(const std::string& csv_file);

/*!
Read a .txt file containing ordered file names.

@param name_file: path to .txt file containing ordered file names. Each line
should represent a unique file. 

@return vector of strings containing file names listed in `name_file`. 
*/
std::vector<std::string> get_file_names(const std::string& name_file);

/*!
Retrieve a vector of images from a vector of file names.

@param file_list: vector of file names.
@param ext: type of file being passed (e.g. ".jpg", ".csv").

@return A vector of images.
*/
std::vector<cv::Mat> file_list_to_data_list(std::vector<std::string> file_list,
                                            const std::string& ext);

/*!
Return recorded centers for bats.

@return vector of points containing centers.
*/
std::vector<cv::Point> read_centers(const std::string& center_file);

/*!
Write vector of images to file.

@param images: vector of images to write.
@param file_names: location to write files to. 

@return void
*/
void img_vec_to_file(std::vector<cv::Mat> images, std::vector<std::string> file_names);


/*!
Calculate edge length for two given centers.

@param center1: first center.
@param center2: second center.

@return distance between the two centers. 
*/
double edge_length(cv::Point center1, cv::Point center2, cv::Point velocity);

/*!
Create an edmonds adjaceny matrix for bipartite object matching.

@param t0_centers: center predictions from the previous time step.
@param t1_centers: center predictions from current time step.

@return 2D vector/matrix representing costs associated with object assignment.
*/
std::vector<std::vector<double> > cost_matrix(std::vector<cv::Point> t0_centers, std::vector<cv::Point> t1_centers, std::vector<cv::Point> velocities);

/*!
Initialize Kalman Filter.

@param first measurement taken.

@return kalman filter for object tracking.
*/

cv::KalmanFilter initialize_kalman(cv::Point first_measure);

/*!
Run prediction and correction events on a Kalman filter to get predicted position.

@param kalman: kalman filter
@param measurement: measured position observation

@return estimated position
*/
cv::Point kalman_loop(cv::KalmanFilter & kalman, cv::Point measurement);

/*!
Draw centers on associated image.

@param img: image where centers are drawn.
@param centers: vector of centers to draw. 

@return void
*/
void draw_centers(cv::Mat & img, std::vector<cv::Point> centers, cv::Scalar color=cv::Scalar(255, 0, 0));

/*!
Draw trajectories of point assignments.

@param img: image where to draw trajectories.
@param trajectories: vector of vectors containing points demarking past positions.

@return void
*/
void draw_trajectories(cv::Mat &img, std::vector<std::vector<cv::Point> > trajectories);

/*!
Initialize vector of vectors to keep track trajectories.

@param t0_centers: centers from time = 0
@param trajectory: vector of vector of points to keep centers for tracked objects.
@param traj_index: maps center point to its associated index in `trajectory`.

@return void
*/
void init_trajectories(std::vector<cv::Point> t0_centers,
                       std::vector<std::vector<cv::Point> >& trajectory,
                       std::map<std::pair<int, int>, int>& traj_index);


/*!
Update trajectory vectors given new assignments.

@param t0_centers: centers from previous frame.
@param t1_centers: centers from current frame.
@param assignments: center assignment from t_0 to t_1. Where the ith index
    represents the ith center from frame 0, and the ith element represents the
    jth element in frame 1 that has been assigned to the previous ith element, t - 1.
@param trajectory: vector of vector of points to keep centers from tracked objects.
@param traj_index: maps center point to its associated index in `trajectory`. 

@return void
*/
void update_trajectories(std::vector<cv::Point> t0_centers,
                         std::vector<cv::Point> t1_centers,
                         std::vector<int> assignments,
                         std::vector<std::vector<cv::Point> >& trajectory,
                         std::map<std::pair<int, int>, int>& traj_index);


/*!
Calculate the velocity of movement between previous points and assigned points.

@param t0_centers: centers from previous assignemnt. 
@param matched_centers: matched centers from current assignment.

@return vector of velocities
*/
std::vector<cv::Point> calculate_velocities(std::vector<cv::Point> t0_centers, std::vector<cv::Point> matched_centers);


/*!
Create a set of indices over a given vector.

Used to determine which centers were assigned by the Hungarian Algorithm.

@param t1_centers: vector of points to iterate over.

@return set of indices
*/
std::set<int> create_index_set(std::vector<cv::Point> t1_centers);

/*!
Add trajectory to tracked trajectories.

@param new_center: new center to initiate trajectory with.
@param trajectories: vector of past trajectories.
@param pair_to_idx: dictionary mapping centers to trajectories.

@return void
*/
void add_trajectory(cv::Point new_center, std::vector<std::vector<cv::Point> >& trajectories, std::map<std::pair<int, int>, int>& pair_to_idx);


/*!
Terminate trajectories by appending (-1, -1) to trajectory vectors.

@param centers: vector of centers pointing to trajectories to terminate
@param trajectories: vector of past trajectories.
@param point_to_traj: dictionary mapping centers to trajectories. 

@return void
*/
void terminate_trajectories(std::vector<cv::Point> centers, std::vector<std::vector<cv::Point> >& trajectories, std::map<std::pair<int, int>, int> & point_to_traj);
    

/*!
Return past centers who are unsimilar to current centers

@param cost_m: cost matrix of center assignment.
@param centers: centers associated with t0.
@param d_thresh: minimum distance between centers to consider. Default is 50.

@return vector of indices for t0 centers that are likely new object.
*/
std::vector<cv::Point> filter_old_centers(std::vector<std::vector<double> > cost_m, std::vector<cv::Point> centers, double d_thresh=50);


/*!
Return new centers who are unsimilar to old centers.

@param cost_m: cost matrix of center assignment.
@param centers: centers associated with t1.
@param d_thresh: minimum distance between centers to consider. Default is 50.
std::vector<cv::Point> filter_old_centers(std::vector<std::vector<double> > cost_m, std::vector<cv::Point> centers, double d_thresh=50);
*/
std::vector<cv::Point> filter_new_centers(std::vector<std::vector<double> > cost_m, std::vector<cv::Point> centers, double d_thresh=50);

/*!
Remove centers from a vector of centers.

@param original: original vector of centers.
@param del_centers: centers to delete from `original`.

@return void
*/
void remove_centers(std::vector<cv::Point>& original, std::vector<cv::Point>& del_centers);

/*!
Calculate velocities from logged trajectories.

@param centers: logged centers.
@param trajectories: vector of trajectories.
@param idx_to_traj: dictionary pointing centers to trajectories. 
*/
std::vector<cv::Point> get_velocities(std::vector<cv::Point> centers, std::vector<std::vector<cv::Point> > trajectories, std::map<std::pair<int, int>, int> idx_to_traj);

/*!
Find unmatched t0 centers.

@param assignments: center assignments from hungarian output.
@param t0_centers: centers from t - 1;

@return vector of unmatched centers.
*/
std::vector<cv::Point> unmatched_t0_centers(std::vector<int> assignments, std::vector<cv::Point> t0_centers);

/*!
Find unmatched t1 centers. 

@param assignments: center assignments from hungarian output.
@param t1_centers: centers from current time.

@return vector of unmatched centers
*/
std::vector<cv::Point> unmatched_t1_centers(std::vector<int> assignments, std::vector<cv::Point> t1_centers);


/*!
Find trajectory associated with given center.

@param center: center of interest.
@param trajectories: vector of trajectories.
@param idx_to_traj: dictionary pointing centers to their trajectories.

@return index associated with center.
*/
int find_trajectory(cv::Point center, std::map<std::pair<int, int>, int>& idx_to_traj);

/*!
Find nearest center in space from a provided center.

@param center: center of interest.
@param centers: centers to search.

@return closest point from `center` in `centers`.
*/
cv::Point find_nearest_center(cv::Point center, std::vector<cv::Point> centers);

/*!
Return a center near provided point that does not have a key in trajectories.

@param center: center of interest.
@param idx_to_traj: dictionary pointing centers to trajectories.

@return poin near `center` that is not in `idx_to_traj`.
*/
cv::Point ensure_unique_center(cv::Point center, std::map<std::pair<int, int>, int>& idx_to_traj);

/*
Update a given trajectory.

@param old_center: point demarking which trajectory to update.
@param new_center: new point to add to trajectory.
@param trajectories: trajectories of observed points.
@param idx_to_traj: dictionary pointing centers to their trajectories.

@return void
*/
void update_trajectory(cv::Point old_center, cv::Point new_center, std::vector<std::vector<cv::Point> >& trajectories, std::map<std::pair<int, int>, int>& idx_to_traj);

/*
Get current tracked points.

@param trajectories: vector of current trajectories.

@return centers whose latest entry in their trajectories is not (-1, -1). 
*/
std::vector<cv::Point> get_current_points(std::vector<std::vector<cv::Point> > trajectories);