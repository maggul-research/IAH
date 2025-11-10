/* ---------------------------------------------------------------------
 * Copyright (C) 2025
 * Authored by Mustafa Aggul
 * Hacettepe University
 * Southern Methodist University
 *
 * This is the implementation file for the qualitative testing of
 * the improved Arrow Hurwicz Method for Natural Convection Equation
 *
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
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
  using namespace std;

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
    mesh_jobs(const unsigned int n_refinements);
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

    vector<types::global_dof_index> dofs_per_block;

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

  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution()
      : Function<dim>(dim + 2)
    {}
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &value) const override;
  };

  template <int dim>
  void
  ExactSolution<dim>::vector_value(const Point<dim> &p,
                                   Vector<double>   &values) const
  {
    const double x = p[0];
    const double y = p[1];

    for (int i = 0; i < dim + 2; i++)
      {
        values[i] = 0.0 * y;
      }
    values[dim + 1] = 1.0 - x;
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
    , solver_type(0) /* GMRES:0 and UMFPACK:1 */
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
  NaturalConvection<dim>::mesh_jobs(const unsigned int n_refinements)
  {
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(n_refinements);

    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->at_boundary())
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->face(f)->at_boundary())
            {
              cell->face(f)->set_boundary_id(0);

              const auto &p = cell->face(f)->center();

              if ((fabs(p(0) - 0.0) < 1e-10 || fabs(p(0) - 1.0) < 1e-10))
                cell->face(f)->set_boundary_id(1);
            }
  }

  template <int dim>
  void
  NaturalConvection<dim>::setup_dofs()
  {
    system_matrix.clear();

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);

    vector<unsigned int> block_component(dim + 2, 0);
    block_component[dim]     = 1;
    block_component[dim + 1] = 2;
    DoFRenumbering::component_wise(dof_handler, block_component);

    dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];
    unsigned int dof_T = dofs_per_block[2];

    FEValuesExtractors::Vector velocities(0);
    FEValuesExtractors::Scalar temperature(dim + 1);
    {
      constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(
                                                 dim + 2),
                                               constraints,
                                               fe.component_mask(velocities));

      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               Functions::ZeroFunction<dim>(
                                                 dim + 2),
                                               constraints,
                                               fe.component_mask(velocities));

      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               ExactSolution<dim>(),
                                               constraints,
                                               fe.component_mask(temperature));
    }
    constraints.close();

    cout << "   Number of active cells: " << triangulation.n_active_cells()
         << endl
         << "   Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
         << dof_u << '+' << dof_p << '+' << dof_T << ')' << endl;
  }

  template <int dim>
  void
  NaturalConvection<dim>::initialize_system()
  {
    {
      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
      sparsity_pattern.copy_from(dsp);
    }

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

    vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    vector<Tensor<1, dim>> velocity_values(n_q_points);
    vector<Tensor<2, dim>> velocity_gradients(n_q_points);
    vector<Tensor<1, dim>> temperature_gradients(n_q_points);
    vector<double>         temperature_values(n_q_points);
    vector<double>         velocity_div(n_q_points);
    vector<double>         pressure_values(n_q_points);

    vector<double>         div_phi_u(dofs_per_cell);
    vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    vector<Tensor<1, dim>> grad_phi_T(dofs_per_cell);
    vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    vector<double>         phi_p(dofs_per_cell);
    vector<double>         phi_T(dofs_per_cell);

    Tensor<1, dim> e_vector;
    e_vector[dim - 1] = 1;

    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                   endc = dof_handler.end();

    for (; cell != endc; ++cell)
      {
        fe_values.reinit(cell);

        local_matrix = 0;
        local_rhs    = 0;

        if (assemble_id == 0)
          {
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
          {
            fe_values[temperature].get_function_values(solution,
                                                       temperature_values);
          }

        if (assemble_id == 2)
          {
            fe_values[velocities].get_function_divergences(solution,
                                                           velocity_div);
          }

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
                    {
                      local_matrix(i, j) +=
                        ((1.0 / rhoU * c1 + Pr) *
                           scalar_product(grad_phi_u[j], grad_phi_u[i])

                         + 1.0 * grad_phi_u[j] * velocity_values[q] * phi_u[i] +
                         0.5 * velocity_div[q] * phi_u[j] * phi_u[i]

                         + (rhoU * c1 / alpha + c2 * beta) * div_phi_u[i] *
                             div_phi_u[j]

                         + (c1 * alpha + c2 / beta) * phi_p[j] * phi_p[i]

                         + (1.0 / rhoT * c3 + 1.0) *
                             scalar_product(grad_phi_T[j], grad_phi_T[i])

                         + 1.0 * grad_phi_T[j] * velocity_values[q] * phi_T[i] +
                         0.5 * velocity_div[q] * phi_T[j] * phi_T[i]

                         ) *
                        fe_values.JxW(q);
                    } // end dof iterations for phi_j

                if (assemble_id == 0)
                  local_rhs(i) +=
                    (1.0 / rhoU * c1 *
                       scalar_product(velocity_gradients[q], grad_phi_u[i])

                     + pressure_values[q] * div_phi_u[i]

                     + (c1 * alpha + c2 / beta) * pressure_values[q] * phi_p[i]

                     +
                     1.0 / rhoT * c3 * temperature_gradients[q] * grad_phi_T[i]

                     ) *
                    fe_values.JxW(q);
                else if (assemble_id == 1)
                  local_rhs(i) += Pr * Ra * temperature_values[q] * e_vector *
                                  phi_u[i] * fe_values.JxW(q);
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
    int blk = 3; // dummy choice 3
    switch (solve_id)
      {
        case 0:
          blk = 2; // solve for temperature corresponds to block 2
          break;
        case 1:
          blk = 0; // solve for velocity corresponds to block 0
          break;
        case 2:
          blk = 1; // solve for pressure corresponds to block 1
          break;
        default:
          cout << " Unknown solve ID " << endl;
          break;
      }

    if (solver_type == 0)
      {
        SolverControl               solver_control(1000000, 1e-6, true);
        SolverGMRES<Vector<double>> gmres(solver_control);

        gmres.solve(system_matrix.block(blk, blk),
                    solution.block(blk),
                    system_rhs.block(blk),
                    PreconditionIdentity());
        cout << " ****GMRES steps: " << solver_control.last_step() << endl;
      }
    else if (solver_type == 1)
      {
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix.block(blk, blk));
        A_direct.vmult(solution.block(blk), system_rhs.block(blk));
      }
    else
      {
        cout << " Unknown solver type " << endl;
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

        current_residue =
          max(residue_vector.block(0).l2_norm() / solution.block(0).l2_norm(),
              residue_vector.block(2).l2_norm() / solution.block(2).l2_norm());

        cout << "******************************" << endl;
        cout << " The relative error of the current iteration = "
             << current_residue << " at " << " at " << th_iteration
             << "th iteration" << endl;
        cout << "******************************" << endl;

        ++th_iteration;

        cout << endl;
        cout << "**"
             << residue_vector.block(0).l2_norm() / solution.block(0).l2_norm()
             << "**"
             << residue_vector.block(1).l2_norm() / solution.block(1).l2_norm()
             << "**"
             << residue_vector.block(2).l2_norm() / solution.block(2).l2_norm()
             << endl;

        if (output_result)
          output_results(th_iteration);
      }
  }

  template <int dim>
  void
  NaturalConvection<dim>::output_results(const unsigned int output_index) const
  {
    vector<string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");
    solution_names.push_back("temperature");

    vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    ostringstream filename;
    filename << "Ra_" << Ra << "_solution_"
             << Utilities::int_to_string(output_index, 4) << ".vtk";

    ofstream output(filename.str().c_str());
    data_out.write_vtk(output);
  }

  template <int dim>
  void
  NaturalConvection<dim>::run(const unsigned int n_refinements)
  {
    mesh_jobs(n_refinements);
    setup_dofs();
    initialize_system();

    run_iterations(1e-4, 10000, true);
  }
} // end namespace MyNSE

int
main()
{
  // input the simulation dimension
  unsigned int dim;
  std::cout << "Enter the simulation dimension (2, 3):\n";
  std::cin >> dim;
  if (dim < 2 || dim > 3)
    {
      std::cout << "Invalid simulation dimension \n";
      return 1;
    }

  // input the number of the algorithm
  unsigned int alg;
  std::cout << "Enter the number of the algorithm (1, 2, 3):\n";
  std::cin >> alg;
  if (alg < 1 || alg > 3)
    {
      std::cout << "Invalid algorithm number \n";
      return 1;
    }
  try
    {
      using namespace dealii;
      using namespace MyNSE;

      Timer timer;

      const unsigned int degree = 1;
      const double       Pr     = 0.71;
      const double       Ra     = 1.0e+6;
      const double       rhoU   = 1.0e-1;
      const double       rhoT   = 2.0e-1;
      const double       alpha  = 1.0e-4;
      const double       beta   = 1000.0;

      if (dim == 2)
        {
          NaturalConvection<2> flow(
            alg, degree, Pr, Ra, rhoU, rhoT, alpha, beta);
          flow.run(6);
        }
      else if (dim == 3)
        {
          NaturalConvection<3> flow(
            alg, degree, Pr, Ra, rhoU, rhoT, alpha, beta);
          flow.run(5);
        }

      timer.stop();
      std::cout << std::endl;
      std::cout
        << "************************************************************"
        << std::endl;
      std::cout << "Elapsed CPU time: " << timer.cpu_time() << " seconds."
                << std::endl;
      std::cout << "Elapsed wall time: " << timer.wall_time() << " seconds."
                << std::endl;
      std::cout
        << "************************************************************"
        << std::endl;
      std::cout << std::endl;
      // reset timer for the next thing it shall do
      timer.reset();
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}