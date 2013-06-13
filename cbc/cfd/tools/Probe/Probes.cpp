#include "Probes.h"

using namespace dolfin;

Probes::Probes(const Array<double>& x, const FunctionSpace& V)
{
  const std::size_t Nd = V.mesh()->geometry().dim();
  const std::size_t N = x.size() / Nd;
  Array<double> _x(Nd);
  total_number_probes = N;
  _value_size = 1;
  _num_evals = 0;
  for (std::size_t i = 0; i < V.element()->value_rank(); i++)
    _value_size *= V.element()->value_dimension(i);

  for (std::size_t i=0; i<N; i++)
  {
    for (std::size_t j=0; j<Nd; j++)
      _x[j] = x[i*Nd + j];
    try
    {
      Probe* probe = new Probe(_x, V);
      std::pair<std::size_t, Probe*> newprobe = std::make_pair(i, &(*probe));
      _allprobes.push_back(newprobe);
    } 
    catch (std::exception &e)
    { // do-nothing
    }
  }
  cout << _allprobes.size() << " of " << N  << " probes found on processor " << MPI::process_number() << endl;
}
//
Probes::~Probes()
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    delete _allprobes[i].second;   
  }
  _allprobes.clear();      
}
//
void Probes::add_positions(const Array<double>& x, const FunctionSpace& V)
{
  const std::size_t gdim = V.mesh()->geometry().dim();
  const std::size_t N = x.size() / gdim;
  Array<double> _x(gdim);
  const std::size_t old_N = total_number_probes;
  total_number_probes += N;

  for (std::size_t i=0; i<N; i++)
  {
    for (std::size_t j=0; j<gdim; j++)
      _x[j] = x[i*gdim + j];
    try
    {
      Probe* probe = new Probe(_x, V);
      std::pair<std::size_t, Probe*> newprobe = std::make_pair(old_N+i, &(*probe));
      _allprobes.push_back(newprobe);
    } 
    catch (std::exception &e)
    { // do-nothing
    }
  }
  cout << _allprobes.size() << " of " << N  << " probes found on processor " << MPI::process_number() << endl;
}
//
void Probes::eval(const Function& u)
{
  u.update();
  for (std::size_t i = 0; i < local_size(); i++)
  {
    _allprobes[i].second->eval(u);
  }
  _num_evals++;
}
//
void Probes::erase_snapshot(std::size_t i)
{
  for (std::size_t j = 0; j < local_size(); j++)
  {
    _allprobes[j].second->erase_snapshot(i);
  }
  _num_evals--;
}
//
void Probes::clear()
{
  for (std::size_t i = 0; i < local_size(); i++)
  {
    _allprobes[i].second->clear();
  }
  _num_evals = 0;
}
//
void Probes::dump(std::size_t i, std::string filename)
{
  std::string ss="";  
  for (std::size_t j = 0; j < local_size(); j++)
  {
    ss = filename;
    ss.append("_");
    ss.append(boost::lexical_cast<std::string>(_allprobes[j].first));
    ss.append(".probe");
    _allprobes[j].second->dump(i, ss, get_probe_id(j));
    ss.clear();
  }
}
//
void Probes::dump(std::string filename)
{
  std::string ss="";  
  for (std::size_t j = 0; j < local_size(); j++)
  {
    ss = filename;
    ss.append("_");
    ss.append(boost::lexical_cast<std::string>(_allprobes[j].first));
    ss.append(".probe");
    _allprobes[j].second->dump(ss, get_probe_id(j));
    ss.clear();
  }
}
//
Probe* Probes::get_probe(std::size_t i)
{
  if (i >= local_size() || i < 0) 
  {
    dolfin_error("Probes.cpp", "get probe", "Wrong index!");
  }
  return _allprobes[i].second;
}
//
std::size_t Probes::get_probe_id(std::size_t i)
{
  if (i >= local_size() || i < 0) 
  {
    dolfin_error("Probes.cpp", "get probe_id", "Wrong index!");
  }
  return _allprobes[i].first;
}
