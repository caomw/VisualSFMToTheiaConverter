#include <theia/theia.h>
#include <glog/logging.h>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>

#include "MatchFile.h"
#include "FeaturePoints.h"

DEFINE_string(images, "", "Wildcard of images to reconstruct.");
DEFINE_string(output_match_file, "", "Location of match file to write.");

// Geometric verification options.
DEFINE_int32(min_num_verified_matches, 30,
            "Minimum number of matches to be considered an inlier.");

Eigen::Matrix3d GetEigen3x3Matrix(const float mat[3][3]) {
  Eigen::Matrix3d eigen_mat;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      eigen_mat(i, j) = mat[i][j];
    }
  }
  return eigen_mat;
}

bool ReadVSFMMatches(const std::string& match_file,
                     const visual_sfm::FeatureData& sift1,
                     visual_sfm::MatchFile* base_image,
                     theia::ImagePairMatch* match) {
  // Load SIFT location data.
  visual_sfm::FeatureData sift2;
  const std::string sift2_file = match_file + ".sift";
  sift2.ReadSIFTB_LOC(sift2_file.c_str());

  // Retrieve matches.
  const int feature_count = sift2.getFeatureNum();
  visual_sfm::Points<int> vsfm_matches;
  visual_sfm::TwoViewGeometry two_view_geometry;
  if (!base_image->GetIMatch(match_file.c_str(),
                             feature_count,
                             two_view_geometry,
                             vsfm_matches)) {
    LOG(INFO) << "Skipping " << match_file << ". No features to match.";
    return false;
  }

  const int num_fmatrix_inliers = two_view_geometry.NF;
  const int num_ematrix_inliers = two_view_geometry.NE;

  const Eigen::Matrix3d fmatrix = GetEigen3x3Matrix(two_view_geometry.F);
  const Eigen::Matrix3d rotation = GetEigen3x3Matrix(two_view_geometry.R);
  const Eigen::Vector3d translation =
      Eigen::Map<Eigen::Vector3f>(two_view_geometry.T).cast<double>();

  const double focal_length1 = two_view_geometry.F1;
  const double focal_length2 = two_view_geometry.F2;

  if (num_ematrix_inliers < FLAGS_min_num_verified_matches ||
      num_fmatrix_inliers < FLAGS_min_num_verified_matches) {
    return false;
  }

  VLOG(1) << "Match info:"
          << "\nNum fmatrix inliers = " << num_fmatrix_inliers
          << "\nNum ematrix inliers = " << num_ematrix_inliers
          << "\nFMatrix = \n" << fmatrix
          << "\nRotation = \n" << rotation
          << "\nTranslation = " << translation.transpose()
          << "\nFocal lengths: " << focal_length1 << ", " << focal_length2;

  // Convert feature matches to Theia types.
  match->correspondences.resize(num_fmatrix_inliers);
  const auto& sift_location1 = sift1.getLocationData();
  const auto& sift_location2 = sift2.getLocationData();
  for (int i = 0; i < num_fmatrix_inliers; i++) {
    match->correspondences[i].feature1 =
        Eigen::Vector2d(sift_location1[vsfm_matches[0][i]][0],
                        sift_location1[vsfm_matches[0][i]][1]);
    match->correspondences[i].feature2 =
        Eigen::Vector2d(sift_location2[vsfm_matches[1][i]][0],
                        sift_location2[vsfm_matches[1][i]][1]);
  }

  // Set params.
  match->twoview_info.focal_length_1 = focal_length1;
  match->twoview_info.focal_length_2 = focal_length2;
  match->twoview_info.position_2 = -rotation.transpose() * translation;
  const Eigen::AngleAxisd rotation_aa(rotation);
  match->twoview_info.rotation_2 = rotation_aa.angle() * rotation_aa.axis();
  match->twoview_info.num_verified_matches = num_fmatrix_inliers;
  return true;
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Get full image filepaths.
  std::vector<std::string> image_files;
  CHECK(theia::GetFilepathsFromWildcard(FLAGS_images, &image_files));
  std::vector<theia::CameraIntrinsicsPrior> intrinsics(image_files.size());

  // Assign an id to all view names.
  std::unordered_map<std::string, int> image_name_to_id;
  std::vector<std::string> pruned_image_files(image_files.size());
  for (int i = 0; i < image_files.size(); i++) {
    // Create the pruned image filename and create an image index.
    std::string filename;
    CHECK(theia::GetFilenameFromFilepath(image_files[i], true, &filename));
    pruned_image_files[i] = filename;

    std::string filename_no_extension;
    CHECK(theia::GetFilenameFromFilepath(image_files[i], false,
                                         &filename_no_extension));
    image_name_to_id[filename_no_extension] = i;
  }

  // Read all putatative matches one by one.
  std::vector<theia::ImagePairMatch> matches;
  std::unordered_set<std::string> visited_images;
  for (int i = 0; i < image_files.size(); i++) {
    // Open the match file (remove the extension).
    const std::string base_filepath =
        image_files[i].substr(0, image_files[i].find_last_of("."));
    const std::string filename1 = pruned_image_files[i];
    visited_images.insert(filename1);

    LOG(INFO) << "Reading matches for image = " << filename1;
    visual_sfm::MatchFile match1(base_filepath.c_str());
    if (!match1.IsValid()) {
      LOG(INFO) << "Could not load matches for " << base_filepath;
      continue;
    }

    // Load the SIFT descriptors.
    visual_sfm::FeatureData sift1;
    const std::string sift1_file = base_filepath + ".sift";
    sift1.ReadSIFTB_LOC(sift1_file.c_str());

    // Inspect all matches.
    std::vector<std::string> matched_images;
    match1.GetMatchedImageList(matched_images);
    for (const std::string& match2 : matched_images) {
      std::string filename2;
      CHECK(theia::GetFilenameFromFilepath(match2, true, &filename2));

      // Do not visit a match twice!
      if (theia::ContainsKey(visited_images, filename2)) {
        continue;
      }

      // Read the match information.
      theia::ImagePairMatch image_pair_match;
      image_pair_match.image1_index = i;
      image_pair_match.image2_index =
          theia::FindOrDie(image_name_to_id, filename2);
      if (ReadVSFMMatches(match2, sift1, &match1, &image_pair_match)) {
        LOG(INFO) << "Matched image " << filename1 << " to image " << filename2
                  << " with " << image_pair_match.correspondences.size()
                  << " inliers.";
        matches.emplace_back(image_pair_match);
      }
    }
  }

  CHECK(theia::WriteMatchesAndGeometry(FLAGS_output_match_file,
                                       pruned_image_files,
                                       intrinsics,
                                       matches))
      << "Could not write match file";

  return 0;
}
