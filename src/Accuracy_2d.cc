/* ---------------------------------------------------------------------
 * Copyright (C) 2025
 * Authored by
 * Mustafa Aggul
 * Hacettepe University
 * Southern Methodist University
 * Sinan Ergen
 * Balikesir University
 *
 * This is the implementation file for the accuracy analysis of
 * the improved Arrow Hurwicz Method for Natural Convection Equation
 *
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace MyNSE
{
  using namespace dealii;

  template <int dim>
  class NaturalConvection
  {
  public:
    NaturalConvection(const unsigned int alg,
                      const unsigned int degree,
                      const double       Pr,
                      const double       Ra,
                      const double       rhoU,
                      const double       rhoT,
                      const double       alpha,
                      const double       beta);
    void
    run(const unsigned int n_refinements);

  private:
    void
    setup_dofs();
    void
    initialize_system();
    /* assemble_id determines what parts of the system will be updated
       assemble_id: 0 assembles the system_matrix and system_rhs
       assemble_id: 1 updates the system_rhs with the current temperature
       assemble_id: 2 updates the system_rhs with the current velocity
    */
    void
    assemble(int assemble_id);
    /* solve_id determines what parts of the system will be solved
       solve_id: 0 solves for the temperature
       solve_id: 1 solves for the velocity
       solve_id: 2 solves for the pressure
    */
    void
    solve(int solve_id);
    void
    output_results(const unsigned int output_index) const;
    void
    compute_errors(BlockVector<double> solution);
    void
    run_iterations(const double       tolerance,
                   const unsigned int max_iteration,
                   const bool         output_result);

    const unsigned int alg;
    const unsigned int degree;
    const double       Pr;
    const double       Ra;
    const double       rhoU;
    const double       rhoT;
    const double       alpha;
    const double       beta;
    const unsigned int solver_type;

    std::vector<types::global_dof_index> dofs_per_block;

    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints;

    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double> old_solution;
    BlockVector<double> solution;
    BlockVector<double> system_rhs;
    BlockVector<double> residue_vector;
  };

  /* Exact solution for the accuracy analysis */
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution()
      : Function<dim>(dim + 2)
    {}
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &value) const override;

  private:
    const double Pr = 1.0;
    const double Ra = 1000.0;
    const double aa = 1.0;
  };

  template <int dim>
  void
  ExactSolution<dim>::vector_value(const Point<dim> &p,
                                   Vector<double>   &values) const
  {
    const double x = p[0];
    const double y = p[1];

    values[0] = aa * pow(y, 3);
    values[1] = aa * pow(x, 3);
    values[2] =
      6.0 * Pr * aa * x * y - 3.0 / 4.0 * pow(aa, 2) * pow(x, 4) * pow(y, 2);
    values[3] = 3.0 * pow(aa, 2) / (2.0 * Ra * Pr) *
                (2.0 * pow(x, 2) * pow(y, 3) - pow(x, 4) * y);
  }

  /* RHS based on the PDE and the given exact solution */
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide()
      : Function<dim>(dim + 2)
    {}
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &value) const override;

  private:
    const double Pr = 1.0;
    const double Ra = 1000.0;
    const double aa = 1.0;
  };
  template <int dim>
  void
  RightHandSide<dim>::vector_value(const Point<dim> &p,
                                   Vector<double>   &values) const
  {
    const double x = p[0];
    const double y = p[1];

    values[0] = 0.0;
    values[1] = 0.0;
    values[2] = 0.0;
    values[3] = (3.0 * pow(aa, 2) * (12.0 * pow(x, 2) * y - 4 * pow(y, 3))) /
                  (2.0 * Pr * Ra) +
                (3.0 * pow(aa, 3) * pow(y, 3) *
                 (-4.0 * pow(x, 3) * y + 4.0 * x * pow(y, 3))) /
                  (2.0 * Pr * Ra) -
                (18.0 * pow(aa, 2) * pow(x, 2) * y) / (Pr * Ra) +
                (3.0 * pow(aa, 3) * pow(x, 3) *
                 (-pow(x, 4) + 6.0 * pow(x, 2) * pow(y, 2))) /
                  (2.0 * Pr * Ra);
  }

  /* Exact velocity gradient for the accuracy analysis */
  template <int dim>
  class ExactVelocityGradient : public Function<dim>
  {
  public:
    ExactVelocityGradient()
      : Function<dim>(dim + 2)
    {}
    virtual void
    vector_gradient(
      const Point<dim>            &p,
      std::vector<Tensor<1, dim>> &gradient_values) const override;

  private:
    const double Pr = 1.0;
    const double Ra = 1000.0;
    const double aa = 1.0;
  };

  template <int dim>
  void
  ExactVelocityGradient<dim>::vector_gradient(
    const Point<dim>            &p,
    std::vector<Tensor<1, dim>> &gradient_values) const
  {
    const double x = p[0];
    const double y = p[1];

    for (unsigned int i = 0; i < dim; i++)
      gradient_values[i].clear();

    gradient_values[0][0] = 0.0;
    gradient_values[0][1] = 3.0 * aa * pow(y, 2);
    gradient_values[1][0] = 3.0 * aa * pow(x, 2);
    gradient_values[1][1] = 0.0;
  }

  /* Exact temperature gradient for the accuracy analysis */
  template <int dim>
  class ExactTemperatureGradient : public Function<dim>
  {
  public:
    ExactTemperatureGradient()
      : Function<dim>(dim + 2)
    {}
    virtual Tensor<1, dim>
    gradient(const Point<dim>  &p,
             const unsigned int component = 0) const override;
  };

  template <int dim>
  Tensor<1, dim>
  ExactTemperatureGradient<dim>::gradient(const Point<dim> &p,
                                          const unsigned int) const
  {
    const double Pr = 1.0;
    const double Ra = 1000.0;
    const double aa = 1.0;

    const double x = p[0];
    const double y = p[1];

    Tensor<1, dim> return_value;

    return_value[0] =
      (3.0 * pow(aa, 2) * (-4.0 * pow(x, 3) * y + 4.0 * x * pow(y, 3))) /
      (2.0 * Pr * Ra);
    return_value[1] =
      (3.0 * pow(aa, 2) * (-pow(x, 4) + 6.0 * pow(x, 2) * pow(y, 2))) /
      (2.0 * Pr * Ra);

    return return_value;
  }

  template <int dim>
  NaturalConvection<dim>::NaturalConvection(const unsigned int alg,
                                            const unsigned int degree,
                                            const double       Pr,
                                            const double       Ra,
                                            const double       rhoU,
                                            const double       rhoT,
                                            const double       alpha,
                                            const double       beta)
    : alg(alg)
    , degree(degree)
    , Pr(Pr)
    , Ra(Ra)
    , rhoU(rhoU)
    , rhoT(rhoT)
    , alpha(alpha)
    , beta(beta)
    , solver_type(1) /* GMRES:0 and UMFPACK:1 */
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , fe(FE_Q<dim>(degree + 1),
         dim,
         FE_Q<dim>(degree),
         1,
         FE_Q<dim>(degree + 1),
         1)
    , dof_handler(triangulation)
  {}

  template <int dim>
  void
  NaturalConvection<dim>::setup_dofs()
  {
    system_matrix.clear();

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);

    std::vector<unsigned int> block_component(dim + 2, 0);
    block_component[dim]     = 1;
    block_component[dim + 1] = 2;
    DoFRenumbering::component_wise(dof_handler, block_component);

    dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];
    const unsigned int n_T = dofs_per_block[2];

    FEValuesExtractors::Vector velocities(0);
    FEValuesExtractors::Scalar temperature(dim + 1);

    constraints.clear();

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             ExactSolution<dim>(),
                                             constraints,
                                             fe.component_mask(velocities));
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             ExactSolution<dim>(),
                                             constraints,
                                             fe.component_mask(temperature));
    constraints.close();

    std::cout << "  Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << '+' << n_T << ')' << std::endl;
  }

  template <int dim>
  void
  NaturalConvection<dim>::initialize_system()
  {
    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    old_solution.reinit(dofs_per_block);
    solution.reinit(dofs_per_block);
    system_rhs.reinit(dofs_per_block);
    residue_vector.reinit(dofs_per_block);

    solution = 0;
  }

  template <int dim>
  void
  NaturalConvection<dim>::assemble(int assemble_id)
  {
    if (assemble_id == 0)
      {
        system_matrix = 0;
        system_rhs    = 0;
      }

    /* Following coefficients ensure
      alg=1 runs algorithm 1
      alg=2 runs algorithm 2
      alg=3 runs algorithm 3
    */
    const int c1 = (3 - alg) * alg / 2;
    const int c2 = (1 - alg) * (2 - alg) / 2;
    const int c3 = (2 - alg) * (3 - alg) / 2;

    QGauss<dim> quadrature_formula(degree + 2);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    const FEValuesExtractors::Scalar temperature(dim + 1);

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> velocity_values(n_q_points);
    std::vector<Tensor<2, dim>> velocity_gradients(n_q_points);
    std::vector<Tensor<1, dim>> temperature_gradients(n_q_points);
    std::vector<double>         temperature_values(n_q_points);
    std::vector<double>         velocity_div(n_q_points);
    std::vector<double>         pressure_values(n_q_points);

    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> grad_phi_T(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);
    std::vector<double>         phi_T(dofs_per_cell);

    Tensor<1, dim> e_vector;
    e_vector[dim - 1] = 1.0;

    RightHandSide<dim>          rhs_fun;
    std::vector<Vector<double>> rhs_values(n_q_points, Vector<double>(dim + 2));

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs    = 0;

        if (assemble_id == 0)
          {
            rhs_fun.vector_value_list(fe_values.get_quadrature_points(),
                                      rhs_values);

            fe_values[velocities].get_function_values(solution,
                                                      velocity_values);
            fe_values[velocities].get_function_gradients(solution,
                                                         velocity_gradients);
            fe_values[velocities].get_function_divergences(solution,
                                                           velocity_div);
            fe_values[temperature].get_function_gradients(
              solution, temperature_gradients);
            fe_values[pressure].get_function_values(solution, pressure_values);
            fe_values[temperature].get_function_values(solution,
                                                       temperature_values);
          }
        if (assemble_id == 1)
          fe_values[temperature].get_function_values(solution,
                                                     temperature_values);
        if (assemble_id == 2)
          fe_values[velocities].get_function_divergences(solution,
                                                         velocity_div);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                if (assemble_id == 0)
                  {
                    div_phi_u[k]  = fe_values[velocities].divergence(k, q);
                    grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                    grad_phi_T[k] = fe_values[temperature].gradient(k, q);
                    phi_u[k]      = fe_values[velocities].value(k, q);
                    phi_p[k]      = fe_values[pressure].value(k, q);
                    phi_T[k]      = fe_values[temperature].value(k, q);
                  }
                if (assemble_id == 1)
                  phi_u[k] = fe_values[velocities].value(k, q);
                if (assemble_id == 2)
                  phi_p[k] = fe_values[pressure].value(k, q);
              }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (assemble_id == 0)
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    local_matrix(i, j) +=
                      (((1.0 / rhoU) * c1 + Pr) *
                         scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                       (grad_phi_u[j] * velocity_values[q]) * phi_u[i] +
                       0.5 * velocity_div[q] * phi_u[j] * phi_u[i] +
                       (rhoU * c1 / alpha + c2 * beta) * div_phi_u[i] *
                         div_phi_u[j] +
                       (c1 * alpha + c2 / beta) * phi_p[j] * phi_p[i] +
                       ((1.0 / rhoT) * c3 + 1.0) *
                         scalar_product(grad_phi_T[j], grad_phi_T[i]) +
                       (grad_phi_T[j] * velocity_values[q]) * phi_T[i] +
                       0.5 * velocity_div[q] * phi_T[j] * phi_T[i]) *
                      fe_values.JxW(q);

                const unsigned int ci = fe.system_to_component_index(i).first;

                if (assemble_id == 0)
                  local_rhs(i) +=
                    ((1.0 / rhoU) * c1 *
                       scalar_product(velocity_gradients[q], grad_phi_u[i]) +
                     pressure_values[q] * div_phi_u[i] +
                     (c1 * alpha + c2 / beta) * pressure_values[q] * phi_p[i] +
                     (1.0 / rhoT) * c3 *
                       (temperature_gradients[q] * grad_phi_T[i]) +
                     fe_values.shape_value(i, q) * rhs_values[q](ci)) *
                    fe_values.JxW(q);
                else if (assemble_id == 1)
                  local_rhs(i) += Pr * Ra * temperature_values[q] *
                                  (e_vector * phi_u[i]) * fe_values.JxW(q);
                else if (assemble_id == 2)
                  local_rhs(i) -= (rhoU * c1 + c2) * velocity_div[q] *
                                  phi_p[i] * fe_values.JxW(q);
              } // end dof iterations for phi_i
          } // end quadrature points iteration

        cell->get_dof_indices(local_dof_indices);
        if (assemble_id == 0)
          constraints.distribute_local_to_global(local_matrix,
                                                 local_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        else
          constraints.distribute_local_to_global(local_rhs,
                                                 local_dof_indices,
                                                 system_rhs);
      } // end cell iteration
  } // end assemble

  template <int dim>
  void
  NaturalConvection<dim>::solve(int solve_id)
  {
    /* temperature: 2, velocity: 0, pressure: 3 */
    int blk = (solve_id == 0 ? 2 : solve_id == 1 ? 0 : solve_id == 2 ? 1 : 3);

    if (solver_type == 0)
      {
        SolverControl               solver_control(100000, 1e-9, true);
        SolverGMRES<Vector<double>> gmres(solver_control);

        gmres.solve(system_matrix.block(blk, blk),
                    solution.block(blk),
                    system_rhs.block(blk),
                    PreconditionIdentity());
        std::cout << " ****GMRES steps: " << solver_control.last_step()
                  << std::endl;
      }
    else if (solver_type == 1)
      {
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix.block(blk, blk));
        A_direct.vmult(solution.block(blk), system_rhs.block(blk));
      }
    else
      {
        std::cout << " Unknown solver type " << std::endl;
      }

    constraints.distribute(solution);
  }

  template <int dim>
  void
  NaturalConvection<dim>::run_iterations(const double       tolerance,
                                         const unsigned int max_iteration,
                                         const bool         output_result)
  {
    unsigned int th_iteration    = 0;
    double       current_residue = 1000.0;

    while ((current_residue > tolerance) && th_iteration < max_iteration)
      {
        old_solution = solution;

        assemble(0);
        solve(0);

        assemble(1);
        solve(1);

        assemble(2);
        solve(2);

        residue_vector = solution;
        residue_vector -= old_solution;

        current_residue = std::max(std::max(residue_vector.block(0).l2_norm() /
                                              solution.block(0).l2_norm(),
                                            residue_vector.block(1).l2_norm() /
                                              solution.block(1).l2_norm()),
                                   residue_vector.block(2).l2_norm() /
                                     solution.block(2).l2_norm());

        std::cout << "******************************" << std::endl;
        std::cout << " The relative error of the current iteration = "
                  << current_residue << " at " << " at " << th_iteration
                  << "th iteration" << std::endl;
        std::cout << "******************************" << std::endl;

        compute_errors(solution);

        ++th_iteration;
      }

    if (output_result)
      output_results(th_iteration);
  }

  template <int dim>
  void
  NaturalConvection<dim>::compute_errors(BlockVector<double> solution)
  {
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + 2);

    const ComponentSelectFunction<dim> pressure_mask(dim, dim + 2);

    const ComponentSelectFunction<dim> temperature_mask(dim + 1, dim + 2);

    Vector<double> cellwise_errors(triangulation.n_active_cells());
    QTrapezoid<1>  q_trapez;
    QIterated<dim> quadrature(q_trapez, 2 * degree + 1 /*degree +1*/);

    ExactSolution<dim> exact_solution;

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &pressure_mask);
    const double current_pressure_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &temperature_mask);
    const double current_temperature_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    const double current_velocity_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    ExactVelocityGradient<dim> exact_solution_gradient;

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution_gradient,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::H1_seminorm,
                                      &velocity_mask);
    const double current_velocity_error_h1 =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    ExactTemperatureGradient<dim> exact_temperature_gradient;

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_temperature_gradient,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::H1_seminorm,
                                      &temperature_mask);
    const double current_temperature_error_h1 =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    std::cout << "Errors: ||e_u||_L2 = " << current_velocity_error
              << ",  ||e_u||_H1 = " << current_velocity_error_h1
              << ",  ||e_T||_L2 = " << current_temperature_error
              << ",  ||e_T||_H1 = " << current_temperature_error_h1
              << ",  ||e_p||_L2 = " << current_pressure_error << std::endl;
    std::cout << std::endl;
  }

  template <int dim>
  void
  NaturalConvection<dim>::output_results(const unsigned int output_index) const
  {
    std::vector<std::string> names(dim, "velocity");
    names.push_back("pressure");
    names.push_back("temperature");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interp(dim, DataComponentInterpretation::component_is_part_of_vector);
    interp.push_back(DataComponentInterpretation::component_is_scalar);
    interp.push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             names,
                             DataOut<dim>::type_dof_data,
                             interp);
    data_out.build_patches();

    std::ostringstream fname;
    fname << "Ra_" << Ra << "_solution_"
          << Utilities::int_to_string(output_index, 4) << ".vtk";
    std::ofstream out(fname.str().c_str());
    data_out.write_vtk(out);
  }

  template <int dim>
  void
  NaturalConvection<dim>::run(const unsigned int n_refinements)
  {
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(n_refinements);

    setup_dofs();
    initialize_system();

    run_iterations(1e-6, 10000, false);
  }
} // end namespace MyNSE

int
main(int argc, char **argv)
{
  try
    {
      using namespace dealii;
      using namespace MyNSE;

      if (argc != 3)
        {
          std::cout << "Usage: " << argv[0]
                    << " <alg_number:1-3> <num_refinements:2-9>\n";
          return 1;
        }

      // read input arguments
      unsigned int alg   = std::atoi(argv[1]);
      unsigned int n_ref = std::atoi(argv[2]);

      // check validity of algorithm number
      if (alg < 1 || alg > 3)
        {
          std::cout << "Invalid algorithm number. Must be 1, 2, or 3.\n";
          return 1;
        }

      // check validity of number of refinements
      if (n_ref < 2 || n_ref > 9)
        {
          std::cout
            << "Invalid number of refinements. Must be between 2 and 9.\n";
          return 1;
        }

      std::cout << "Algorithm: " << alg << "\n";
      std::cout << "Number of refinements: " << n_ref << "\n";
      const unsigned int degree = 1;
      const double       Pr     = 1.0;
      const double       Ra     = 1000.0;
      const double       rhoU   = 100.0;
      const double       rhoT   = 10.0;
      const double       alpha  = 0.1;
      const double       beta   = 1000.0;

      NaturalConvection<2> flow(alg, degree, Pr, Ra, rhoU, rhoT, alpha, beta);
      flow.run(n_ref);
    }
  catch (std::exception &exc)
    {
      std::cerr << "\n\n----------------------------------------------------\n";
      std::cerr << "Exception: " << exc.what() << "\nAborting!\n";
      std::cerr << "----------------------------------------------------\n";
      return 1;
    }
  catch (...)
    {
      std::cerr << "\n\n----------------------------------------------------\n";
      std::cerr << "Unknown exception!\nAborting!\n";
      std::cerr << "----------------------------------------------------\n";
      return 1;
    }
  return 0;
}
