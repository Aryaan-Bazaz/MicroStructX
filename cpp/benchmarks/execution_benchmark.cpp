#include "microstructx/csv_loader.hpp"
#include "microstructx/execution_core.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::vector<microstructx::BboEvent> generate_events(std::size_t n_events) {
    std::vector<microstructx::BboEvent> events;
    events.reserve(n_events);

    std::mt19937_64 rng(7);
    std::normal_distribution<double> ret_dist(0.0, 0.00035);
    std::uniform_real_distribution<double> size_dist(50.0, 500.0);
    double mid = 100.0;

    for (std::size_t idx = 0; idx < n_events; ++idx) {
        mid *= std::exp(ret_dist(rng));
        const double spread = mid * 0.00012;
        const double bid_size = size_dist(rng);
        const double ask_size = size_dist(rng);
        events.push_back(microstructx::BboEvent{
            static_cast<std::int64_t>(idx),
            mid,
            mid - spread / 2.0,
            mid + spread / 2.0,
            bid_size,
            ask_size,
            (bid_size + ask_size) / 20.0,
        });
    }

    return events;
}

std::vector<microstructx::Order> generate_orders(std::size_t n_orders, std::size_t n_events) {
    std::vector<microstructx::Order> orders;
    orders.reserve(n_orders);

    std::mt19937_64 rng(11);
    std::uniform_int_distribution<std::int64_t> ts_dist(0, static_cast<std::int64_t>(n_events - 3));
    std::uniform_real_distribution<double> qty_dist(1.0, 120.0);

    for (std::size_t idx = 0; idx < n_orders; ++idx) {
        orders.push_back(microstructx::Order{
            ts_dist(rng),
            idx % 2 == 0 ? microstructx::Side::Buy : microstructx::Side::Sell,
            idx % 3 == 0 ? microstructx::OrderType::Limit : microstructx::OrderType::Market,
            qty_dist(rng),
        });
    }

    return orders;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t n_events = 1'000'000;
    std::size_t n_orders = 1'000'000;
    std::string csv_path;

    for (int idx = 1; idx < argc; ++idx) {
        const std::string arg = argv[idx];
        if (arg == "--events" && idx + 1 < argc) {
            n_events = static_cast<std::size_t>(std::strtoull(argv[++idx], nullptr, 10));
        } else if (arg == "--orders" && idx + 1 < argc) {
            n_orders = static_cast<std::size_t>(std::strtoull(argv[++idx], nullptr, 10));
        } else if (arg == "--csv" && idx + 1 < argc) {
            csv_path = argv[++idx];
        }
    }

    std::vector<microstructx::BboEvent> events = csv_path.empty()
        ? generate_events(n_events)
        : microstructx::load_bbo_csv(csv_path, n_events);
    if (events.size() < 3) {
        std::cerr << "Need at least 3 BBO events\n";
        return 1;
    }

    auto orders = generate_orders(n_orders, events.size());
    const microstructx::ExecutionConfig config;

    const auto start = std::chrono::high_resolution_clock::now();
    const auto fills = microstructx::execute_orders(events, orders, config);
    const auto end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> elapsed = end - start;
    double total_pnl = 0.0;
    double filled_qty = 0.0;
    for (const auto& fill : fills) {
        total_pnl += fill.realized_pnl;
        filled_qty += fill.filled_qty;
    }

    const double orders_per_second = static_cast<double>(fills.size()) / elapsed.count();
    std::cout << "MicroStructX C++ execution benchmark\n";
    std::cout << "events=" << events.size() << "\n";
    std::cout << "orders=" << orders.size() << "\n";
    std::cout << "fills=" << fills.size() << "\n";
    std::cout << "elapsed_seconds=" << elapsed.count() << "\n";
    std::cout << "orders_per_second=" << orders_per_second << "\n";
    std::cout << "filled_qty=" << filled_qty << "\n";
    std::cout << "realized_pnl=" << total_pnl << "\n";

    return 0;
}
