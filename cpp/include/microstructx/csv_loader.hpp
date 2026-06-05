#pragma once

#include "microstructx/execution_core.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace microstructx {

std::vector<BboEvent> load_bbo_csv(const std::string& path, std::size_t max_rows = 0);

}  // namespace microstructx
