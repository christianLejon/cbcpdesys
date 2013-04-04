#ifndef __STATISTICSPROBES_H
#define __STATISTICSPROBES_H

#include "Probes.h"
#include "StatisticsProbe.h"

namespace dolfin
{
  
  class StatisticsProbes : public Probes
  {
      
  public:
      
    StatisticsProbes(const Array<double>& x, const FunctionSpace& V, bool segregated=false);

    // Return an instance of probe i
    StatisticsProbe* get_probe(std::size_t i);
    
    // For regular and segregated velocity components 
    void eval(const Function& u);
    void eval(const Function& u, const Function& v); // 2D segregated
    void eval(const Function& u, const Function& v, const Function& w); // 3D segregated
    
    // No snapshots for statistics, just averages.
    void erase_snapshot(std::size_t i) {cout << "Cannot erase snapshot for StatisticsProbes" << endl;};
    
  };
}

#endif
