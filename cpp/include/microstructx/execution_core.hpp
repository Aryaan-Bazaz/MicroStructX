#pragma once

#include <cstdint>
#include <vector>

namespace microstructx {

enum class Side : std::uint8_t {
    Buy,
    Sell,
};

enum class OrderType : std::uint8_t {
    Market,
    Limit,
};

struct BboEvent {
    std::int64_t timestamp{};
    double mid_price{};
    double bid_price{};
    double ask_price{};
    double bid_size{};
    double ask_size{};
    double volume{};
};

struct Order {
    std::int64_t timestamp{};
    Side side{Side::Buy};
    OrderType type{OrderType::Market};
    double quantity{};
};

struct ExecutionConfig {
    double fee_bps{0.5};
    double temporary_impact_bps{2.0};
    double permanent_impact_bps{0.2};
    double max_participation_rate{0.25};
    std::int64_t latency_events{1};
    double min_limit_fill_probability{0.02};
    double queue_ahead_fraction{0.28};
    double latency_queue_penalty{0.12};
    double cancellation_rate{0.12};
};

struct FillResult {
    std::int64_t timestamp{};
    double requested_qty{};
    double filled_qty{};
    double fill_price{};
    double notional{};
    double fees{};
    double slippage{};
    double queue_pressure{};
    double queue_position_ahead{};
    double queue_depletion{};
    double limit_fill_probability{};
    double realized_pnl{};
};

FillResult execute_order(
    const BboEvent& event,
    const BboEvent& next_event,
    const Order& order,
    const ExecutionConfig& config
);

std::vector<FillResult> execute_orders(
    const std::vector<BboEvent>& events,
    const std::vector<Order>& orders,
    const ExecutionConfig& config
);

}  // namespace microstructx
