#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <ctime>

#include "particle_filter.h"
#include "Eigen/Dense"

#define EPS 0.00001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles. Initialize all particles to first position (based on estimates of
  // x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  if (is_initialized){
    return;
  }

  num_particles = 200;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  double sample_x, sample_y, sample_theta;

  for (unsigned int i=0; i<num_particles; i++){

    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);
    Particle par = {
        (int) i,  //ID
        sample_x, //x
        sample_y, //y
        sample_theta, //theta
        1.0  //weight
    };

    particles.push_back(par);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.

  double yaw_delta = yaw_rate*delta_t;
  double vel_yaw = velocity/yaw_rate;

  for (unsigned int i=0; i<num_particles; i++){
    Particle &par = particles[i];
    double new_x, new_y, new_theta;

    // Check for constant yaw_rate
    if (fabs(yaw_rate) < EPS) {
      new_x = par.x + velocity * delta_t * cos(par.theta);
      new_y = par.y + velocity * delta_t * sin(par.theta);
      new_theta = par.theta;
    }
    else {
      new_x = par.x + vel_yaw*(sin(par.theta + yaw_delta) - sin(par.theta));
      new_y = par.y + vel_yaw*(cos(par.theta) - cos(par.theta + yaw_delta));
      new_theta = par.theta + yaw_delta;
    }

    normal_distribution<double> dist_x(new_x, std_pos[0]);
    normal_distribution<double> dist_y(new_y, std_pos[1]);
    normal_distribution<double> dist_theta(new_theta, std_pos[2]);

    par.x = dist_x(gen);
    par.y = dist_y(gen);
    par.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations,
  double std_landmark[]) {
  // Find the predicted measurement that is closest to each observed measurement and assign the
  // observed measurement to this particular landmark.

  MatrixXd S = MatrixXd(2,2);
  S << std_landmark[0], 0,
     0, std_landmark[1];
  MatrixXd S_inv = S.inverse();

  double diff_x, diff_y, new_x, new_y;
  // Calculate Mahalanobis distance from each observation to every landmark
  for (unsigned int i=0; i<observations.size(); i++){
    std::vector<double> mahalanobis;
    for (unsigned int j=0; j<predicted.size(); j++){
      diff_x = predicted[j].x - observations[i].x;
      diff_y = predicted[j].y - observations[i].y;
      new_x = diff_x*S_inv(0,0) + diff_y*S_inv(1,0);
      new_y = diff_x*S_inv(0,1) + diff_y*S_inv(1,1);
      mahalanobis.push_back(new_x*diff_x + new_y*diff_y);
    }
    // Find position of min value
    std::vector<double>::iterator result = std::min_element(mahalanobis.begin(), mahalanobis.end());
    unsigned int minIdx = std::distance(mahalanobis.begin(), result);
    observations[i].id = predicted[minIdx].id;
  }

  // Code below is the vectorized version of Mahalanobis. Unfortunately creating
  // the initial matrices from vectors makes it too slow.

  /*
  MatrixXd S = MatrixXd(2,2);
  S << std_landmark[0], 0,
     0, std_landmark[1];
  MatrixXd S_inv = S.inverse();

  MatrixXd predicted_mat = MatrixXd(2, predicted.size());
  MatrixXd observations_mat = MatrixXd(2, observations.size());

  // Transform to matrices to make use of vectorized operations
  for (unsigned int i=0; i<predicted.size(); i++){
    predicted_mat.col(i) << predicted[i].x, predicted[i].y;
  }
  for (unsigned int i=0; i<observations.size(); i++){
    observations_mat.col(i) << observations[i].x, observations[i].y;
  }

  // Calculate Mahalanobis distance from each observation to every landmark
  for (unsigned int i=0; i<observations.size(); i++){
    MatrixXd diff = predicted_mat.colwise() - observations_mat.col(i);
    MatrixXd prod = diff.transpose() * S_inv;
    // Eigen will enforce lazy evaluation using diagonal() and will only calculate ith row * ith column
    // as explained here: https://stackoverflow.com/questions/37863668/eigen-use-of-diagonal-matrix
    // Also notice there is no need to sqrt() the results since sqrt is monotonic increasing
    // we will do it for the sake of completeness, but it is not necessary
    VectorXd mahalanobis = (prod * diff).diagonal().array().sqrt();

    VectorXd::Index minIdx;
    (void) mahalanobis.minCoeff(&minIdx); // Ignore output value
    observations[i].id = predicted[minIdx].id;
  }
  */

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // Update the weights of each particle using a multi-variate Gaussian distribution.
  // Note: The observations are given in the VEHICLE'S coordinate system. The particles are located
  // according to the MAP'S coordinate system

  double std_land_x = std_landmark[0];
  double std_land_y = std_landmark[1];
  double one_over2pi = 1.0/(2.0*M_PI*std_land_x*std_land_y);
  double two_std_x_squared = 2.0*std_land_x*std_land_x;
  double two_std_y_squared = 2.0*std_land_y*std_land_y;

  for (unsigned int i=0; i<num_particles; i++){
    std::vector<LandmarkObs> predictions;
    double par_x = particles[i].x;
    double par_y = particles[i].y;
    double par_theta = particles[i].theta;
    // Find landmarks in particle's range
    for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++){
      int land_id = map_landmarks.landmark_list[j].id_i;
      double land_x = map_landmarks.landmark_list[j].x_f;
      double land_y = map_landmarks.landmark_list[j].y_f;
      if (dist(par_x, par_y, land_x, land_y) < sensor_range){
        predictions.push_back(LandmarkObs{ land_id, land_x, land_y });
      }
    }

    // Create transformation matrix
    MatrixXd T = MatrixXd(3, 3);
    double cos_theta = cos(par_theta);
    double sin_theta = sin(par_theta);
    T << cos_theta, -sin_theta, par_x,
       sin_theta, cos_theta, par_y,
       0, 0, 1;

    std::vector<LandmarkObs> trans_observations;
    // Could vectorize transformation by creating a matrix with all observations
    // problem is we will still need 2 for loops, one for initialization of the matrix
    // and another one to get the results from the matrix in the appropriate struct container
    for (unsigned int j=0; j<observations.size(); j++){
      VectorXd obser_vec = VectorXd(3);
      obser_vec << observations[j].x, observations[j].y, 1;
      // Transform observations from particle's coordinate to map coordinate system
      VectorXd transformed = T * obser_vec;
      trans_observations.push_back(LandmarkObs{ observations[j].id, transformed(0), transformed(1)});
    }

    // Associate observations with landmarks
    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);
    dataAssociation(predictions, trans_observations, std_landmark);
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    //cout<<"Time elapsed: "<<elapsed<<endl;

    // Update weight
    particles[i].weight = 1.0;
    for (unsigned int j=0; j<trans_observations.size(); j++){
      double obs_x = trans_observations[j].x;
      double obs_y = trans_observations[j].y;
      double obs_id = trans_observations[j].id;
      double land_x;
      double land_y;

      // Find associated landmark
      bool found = false;
      unsigned int k = 0;
      while (!found && k<predictions.size()){
        if (predictions[k].id == obs_id){
          land_x = predictions[k].x;
          land_y = predictions[k].y;
          found = true;
        }
        k++;
      }

      double x_diff = obs_x - land_x;
      double y_diff = obs_y - land_y;

      particles[i].weight *= one_over2pi * exp(-(x_diff*x_diff/two_std_x_squared + y_diff*y_diff/two_std_y_squared));
    }
  }

}

void ParticleFilter::resample() {

  std::vector<Particle> new_particles;
  std::vector<double> weights;
  for (unsigned int i=0; i<num_particles; i++){
    weights.push_back(particles[i].weight);
  }
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());

  for (unsigned int i=0; i<num_particles; i++){
    int index = distribution(gen);
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
