#include "rhoban_bbo/optimizer.h"

namespace rhoban_bbo
{
class CrossEntropy : public Optimizer
{
public:
  CrossEntropy();
  CrossEntropy(const CrossEntropy& other);

  Eigen::VectorXd train(RewardFunc& reward, const Eigen::VectorXd& initial_candidate,
                        std::default_random_engine* engine) override;

  /// Does not influence the number of generations, best_set_size is always 10% of pop_size
  virtual void setMaxCalls(int max_calls) override;

  /// Initial covariance matrix is built upon the parameters limits
  Eigen::MatrixXd getInitialCovariance();

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

  virtual std::unique_ptr<Optimizer> clone() const override;

  typedef std::pair<Eigen::VectorXd, double> ScoredCandidate;

private:
  /// How many generations are used
  int nb_generations;

  /// How many samples are generated at each iteration
  int population_size;

  /// How many samples are considered for the next generation
  int best_set_size;

  /// When determining the initial covariance, can be used to start with a lower covariance
  double dev_prescaler;

  /// 0 : Silent
  /// 1+: Write the new parameters when 'setMaxCalls' is called
  int verbosity;
};

}  // namespace rhoban_bbo
