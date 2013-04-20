#include "StatisticsProbes.h"

using namespace dolfin;

StatisticsProbes::StatisticsProbes(const Array<double>& x, const FunctionSpace& V, bool segregated)
{
  const std::size_t Nd = V.mesh()->geometry().dim();
  const std::size_t N = x.size() / Nd;
  Array<double> _x(Nd);
  total_number_probes = N;
  _num_evals = 0;
  _value_size = 1;
  for (std::size_t i = 0; i < V.element()->value_rank(); i++)
    _value_size *= V.element()->value_dimension(i);
  
  if (segregated)
  {
    assert(V.element()->value_rank() == 0);
    _value_size *= V.element()->geometric_dimension();
  }
    
  // Symmetric statistics. Velocity: u, v, w, uu, vv, ww, uv, uw, vw
  _value_size = _value_size*(_value_size+3)/2.;

  for (std::size_t i=0; i<N; i++)
  {
    for (std::size_t j=0; j<Nd; j++)
      _x[j] = x[i*Nd + j];
    try
    {
      StatisticsProbe* probe = new StatisticsProbe(_x, V, segregated);
      std::pair<std::size_t, StatisticsProbe*> newprobe = std::make_pair(i, &(*probe));
      _allprobes.push_back(newprobe);
    } 
    catch (std::exception &e)
    { // do-nothing
    }
  }
  cout << local_size() << " of " << N  << " probes found on processor " << MPI::process_number() << endl;
}
//
StatisticsProbe* StatisticsProbes::get_probe(std::size_t i)
{
  if (i >= local_size() || i < 0) 
  {
    dolfin_error("StatisticsProbes.cpp", "get probe", "Wrong index!");
  }
  StatisticsProbe* probe = (StatisticsProbe*) _allprobes[i].second;
  return probe;
}
//
void StatisticsProbes::eval(const Function& u)
{
  u.update();
  for (std::size_t i = 0; i < local_size(); i++)
  {
    StatisticsProbe* probe = (StatisticsProbe*) _allprobes[i].second;
    probe->eval(u);
  }
  _num_evals++;
}
void StatisticsProbes::eval(const Function& u, const Function& v)
{
  u.update();
  v.update();
  for (std::size_t i = 0; i < local_size(); i++)
  {
    StatisticsProbe* probe = (StatisticsProbe*) _allprobes[i].second;
    probe->eval(u, v);
  }
  _num_evals++;    
}
void StatisticsProbes::eval(const Function& u, const Function& v, const Function& w)
{
  u.update();
  v.update();
  w.update();
  for (std::size_t i = 0; i < local_size(); i++)
  {
    StatisticsProbe* probe = (StatisticsProbe*) _allprobes[i].second;
    probe->eval(u, v, w);
  }
  _num_evals++;
}
void StatisticsProbes::clear()
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    StatisticsProbe* probe = (StatisticsProbe*) _allprobes[i].second;
    probe->clear();
  }
  _num_evals = 0;
}
//
