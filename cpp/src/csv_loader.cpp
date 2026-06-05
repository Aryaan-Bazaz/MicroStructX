#include "microstructx/csv_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace microstructx {
namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

bool looks_like_header(const std::vector<std::string>& fields) {
    if (fields.empty()) {
        return false;
    }
    try {
        static_cast<void>(std::stod(fields.front()));
        return false;
    } catch (...) {
        return true;
    }
}

BboEvent parse_canonical(const std::vector<std::string>& fields) {
    if (fields.size() < 6) {
        throw std::runtime_error("canonical BBO CSV needs at least 6 columns");
    }
    return BboEvent{
        static_cast<std::int64_t>(std::stoll(fields[0])),
        std::stod(fields[1]),
        std::stod(fields[2]),
        std::stod(fields[3]),
        std::stod(fields[4]),
        std::stod(fields[5]),
        fields.size() > 7 ? std::stod(fields[7]) : (std::stod(fields[4]) + std::stod(fields[5])) / 20.0,
    };
}

BboEvent parse_binance_bookticker(const std::vector<std::string>& fields) {
    if (fields.size() < 7) {
        throw std::runtime_error("Binance bookTicker CSV needs 7 columns");
    }
    const double bid_price = std::stod(fields[1]);
    const double bid_qty = std::stod(fields[2]);
    const double ask_price = std::stod(fields[3]);
    const double ask_qty = std::stod(fields[4]);
    return BboEvent{
        static_cast<std::int64_t>(std::stoll(fields[6])),
        (bid_price + ask_price) / 2.0,
        bid_price,
        ask_price,
        bid_qty,
        ask_qty,
        (bid_qty + ask_qty) / 20.0,
    };
}

bool is_valid(const BboEvent& event) {
    return event.mid_price > 0.0
        && event.bid_price > 0.0
        && event.ask_price > 0.0
        && event.bid_size >= 0.0
        && event.ask_size >= 0.0;
}

}  // namespace

std::vector<BboEvent> load_bbo_csv(const std::string& path, std::size_t max_rows) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("could not open BBO CSV: " + path);
    }

    std::vector<BboEvent> events;
    events.reserve(max_rows == 0 ? 1024 : max_rows);

    std::string line;
    bool first_line = true;
    bool canonical = true;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        const auto fields = split_csv_line(line);
        if (first_line) {
            first_line = false;
            if (looks_like_header(fields)) {
                canonical = line.find("mid_price") != std::string::npos;
                continue;
            }
            canonical = fields.size() != 7;
        }

        const BboEvent event = canonical ? parse_canonical(fields) : parse_binance_bookticker(fields);
        if (is_valid(event)) {
            events.push_back(event);
        }
        if (max_rows != 0 && events.size() >= max_rows) {
            break;
        }
    }

    return events;
}

}  // namespace microstructx
