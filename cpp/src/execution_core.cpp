#include "microstructx/execution_core.hpp"

#include <algorithm>
#include <cmath>

namespace microstructx {
namespace {

double clamp(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

double side_sign(Side side) {
    return side == Side::Buy ? 1.0 : -1.0;
}

}  // namespace

FillResult execute_order(
    const BboEvent& event,
    const BboEvent& next_event,
    const Order& order,
    const ExecutionConfig& config
) {
    const double sign = side_sign(order.side);
    const double requested_qty = std::abs(order.quantity);
    const bool is_buy = order.side == Side::Buy;

    const double opposite_depth = is_buy ? event.ask_size : event.bid_size;
    const double same_side_depth = is_buy ? event.bid_size : event.ask_size;
    const double touch_price = is_buy ? event.ask_price : event.bid_price;
    const double queue_pressure = clamp(requested_qty / (opposite_depth + 1.0), 0.0, 10.0);

    const double queue_position_ahead =
        same_side_depth
        * config.queue_ahead_fraction
        * (1.0 + static_cast<double>(config.latency_events) * config.latency_queue_penalty);
    const double queue_cancellations = same_side_depth * config.cancellation_rate;

    const double imbalance_numerator = is_buy
        ? (-event.bid_size + event.ask_size)
        : (event.bid_size - event.ask_size);
    const double imbalance_term = imbalance_numerator / (event.bid_size + event.ask_size + 1.0);
    const double queue_depletion =
        std::max(0.0, event.volume * (1.0 + imbalance_term * 0.35) + queue_cancellations);

    double limit_fill_probability = 0.0;
    if (queue_depletion > queue_position_ahead) {
        limit_fill_probability = clamp(
            (queue_depletion - queue_position_ahead) / (requested_qty + 1.0),
            config.min_limit_fill_probability,
            1.0
        );
    }

    const double max_fill_qty = opposite_depth * config.max_participation_rate;
    double filled_qty = 0.0;
    if (order.type == OrderType::Market) {
        filled_qty = std::min(requested_qty, max_fill_qty);
    } else {
        const double fillable_after_queue = std::max(0.0, queue_depletion - queue_position_ahead);
        filled_qty = std::min({requested_qty, max_fill_qty, fillable_after_queue}) * limit_fill_probability;
    }

    const double temporary_impact =
        sign * event.mid_price * (config.temporary_impact_bps / 10'000.0) * queue_pressure;
    const double fill_price = touch_price + temporary_impact;
    const double notional = filled_qty * fill_price;
    const double fees = notional * (config.fee_bps / 10'000.0);
    const double slippage = sign * filled_qty * (fill_price - event.mid_price);
    const double realized_pnl = sign * filled_qty * (next_event.mid_price - fill_price) - fees;

    return FillResult{
        order.timestamp,
        requested_qty,
        filled_qty,
        fill_price,
        notional,
        fees,
        slippage,
        queue_pressure,
        queue_position_ahead,
        queue_depletion,
        limit_fill_probability,
        realized_pnl,
    };
}

std::vector<FillResult> execute_orders(
    const std::vector<BboEvent>& events,
    const std::vector<Order>& orders,
    const ExecutionConfig& config
) {
    std::vector<FillResult> fills;
    fills.reserve(orders.size());

    for (const auto& order : orders) {
        const std::int64_t event_index = order.timestamp + config.latency_events;
        if (event_index < 0 || static_cast<std::size_t>(event_index) >= events.size()) {
            continue;
        }

        const auto& event = events[static_cast<std::size_t>(event_index)];
        const auto& next_event = static_cast<std::size_t>(event_index + 1) < events.size()
            ? events[static_cast<std::size_t>(event_index + 1)]
            : event;
        fills.push_back(execute_order(event, next_event, order, config));
    }

    return fills;
}

}  // namespace microstructx
