#include "rhoban_bbo/cross_entropy.h"

#include "rhoban_random/multivariate_gaussian.h"
#include <rhoban_utils/util.h>

#include <iostream>

namespace rhoban_bbo
{
bool candidateSort(const CrossEntropy::ScoredCandidate& c1, const CrossEntropy::ScoredCandidate& c2)
{
  return c1.second > c2.second;
}

CrossEntropy::CrossEntropy()
  : nb_generations(10), population_size(100), best_set_size(10), dev_prescaler(1.0), verbosity(0)
{
}

CrossEntropy::CrossEntropy(const CrossEntropy& other)
  : nb_generations(other.nb_generations)
  , population_size(other.population_size)
  , best_set_size(other.best_set_size)
  , dev_prescaler(other.dev_prescaler)
  , verbosity(other.verbosity)
{
}

Eigen::VectorXd CrossEntropy::train(RewardFunc& reward_sampler, const Eigen::VectorXd& initial_candidate,
                                    std::default_random_engine* engine)
{
  Eigen::VectorXd mean = initial_candidate;
  Eigen::MatrixXd covar = getInitialCovariance();
  Eigen::MatrixXd limits = getLimits();

  for (int generation = 0; generation < nb_generations; generation++)
  {
    // Getting samples of the generation
    rhoban_random::MultivariateGaussian distrib(mean, covar);
    Eigen::MatrixXd samples = distrib.getSamples(population_size, engine);
    // Scoring samples
    std::vector<ScoredCandidate> candidates(population_size);
    for (int sample_id = 0; sample_id < population_size; sample_id++)
    {
      Eigen::VectorXd bounded_sample = samples.col(sample_id);
      for (int dim = 0; dim < bounded_sample.rows(); dim++)
      {
        double original = bounded_sample(dim);
        double min = limits(dim, 0);
        double max = limits(dim, 1);
        bounded_sample(dim) = std::min(max, std::max(min, original));
      }
      candidates[sample_id].first = bounded_sample;
      candidates[sample_id].second = reward_sampler(bounded_sample, engine);
    }
    // Sorting candidates
    std::sort(candidates.begin(), candidates.end(), candidateSort);
    // Picking best samples
    Eigen::MatrixXd best_samples(initial_candidate.rows(), best_set_size);
    for (int i = 0; i < best_set_size; i++)
    {
      best_samples.col(i) = candidates[i].first;
    }
    // Updating mean and covariance matrix
    mean = best_samples.rowwise().mean();
    Eigen::MatrixXd centered = best_samples.colwise() - mean;
    covar = (centered * centered.adjoint()) / double(best_samples.cols());
  }
  return mean;
}

Eigen::MatrixXd CrossEntropy::getInitialCovariance()
{
  Eigen::MatrixXd limits = getLimits();
  Eigen::MatrixXd init_covar = Eigen::MatrixXd::Zero(limits.rows(), limits.rows());
  for (int dim = 0; dim < limits.rows(); dim++)
  {
    double amplitude = limits(dim, 1) - limits(dim, 0);
    // In a normal distribution, 95% of the values are in:
    // [mu - 1.96 stddev, mu + 1.96 stddev]
    double dev = dev_prescaler * amplitude / (2 * 1.96);
    init_covar(dim, dim) = dev * dev;
  }
  return init_covar;
}

void CrossEntropy::setMaxCalls(int max_calls)
{
  nb_generations = std::max((int)log2(max_calls), 2);
  population_size = (int)(max_calls / nb_generations);
  best_set_size = std::max(2, (int)(population_size / 10));  // Keeping the 10% best
  if (best_set_size >= population_size)
  {
    throw std::logic_error("CrossEntropy::setMaxCalls: max_calls is too low");
  }
  if (verbosity >= 1)
  {
    std::cout << DEBUG_INFO << "{nb generations:" << nb_generations << ", population size:" << population_size
              << ", best set size:" << best_set_size << "}" << std::endl;
  }
}

std::string CrossEntropy::getClassName() const
{
  return "CrossEntropy";
}

Json::Value CrossEntropy::toJson() const
{
  Json::Value v;
  v["nb_generations"] = nb_generations;
  v["population_size"] = population_size;
  v["best_set_size"] = best_set_size;
  v["dev_prescaler"] = dev_prescaler;
  v["verbosity"] = verbosity;
  return v;
}

void CrossEntropy::fromJson(const Json::Value& v, const std::string& dir_name)
{
  (void)dir_name;
  rhoban_utils::tryRead(v, "nb_generations", &nb_generations);
  rhoban_utils::tryRead(v, "population_size", &population_size);
  rhoban_utils::tryRead(v, "best_set_size", &best_set_size);
  rhoban_utils::tryRead(v, "dev_prescaler", &dev_prescaler);
  rhoban_utils::tryRead(v, "verbosity", &verbosity);

  if (dev_prescaler <= 0.0 || dev_prescaler > 1.0)
  {
    throw std::runtime_error(DEBUG_INFO + "Invalid value for dev_prescaler" + std::to_string(dev_prescaler) + " ]0,1]");
  }
}

std::unique_ptr<Optimizer> CrossEntropy::clone() const
{
  return std::unique_ptr<Optimizer>(new CrossEntropy(*this));
}

}  // namespace rhoban_bbo
