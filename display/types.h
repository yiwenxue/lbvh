#pragma once

#include <cstdint>

struct Extent2D {
    Extent2D() = default;

    Extent2D(const Extent2D &) = default;

    inline Extent2D(std::uint32_t width, std::uint32_t height) : width{width}, height{height} {}

    Extent2D operator+(const Extent2D &other) const {
        return {this->width + other.width, this->height + other.height};
    }

    Extent2D operator-(const Extent2D &other) const {
        return {this->width - other.width, this->height - other.height};
    }

    std::uint32_t width  = 0;
    std::uint32_t height = 0;
};

struct Extent3D {
    Extent3D()                 = default;
    Extent3D(const Extent3D &) = default;

    inline Extent3D(std::uint32_t width, std::uint32_t height, std::uint32_t depth) : width{width}, height{height}, depth{depth} {}

    Extent3D operator+(const Extent3D &other) const {
        return {this->width + other.width, this->height + other.height, this->depth + other.depth};
    }

    Extent3D operator-(const Extent3D &other) const {
        return {this->width - other.width, this->height - other.height, this->depth + other.depth};
    }

    std::uint32_t width  = 0;
    std::uint32_t height = 0;
    std::uint32_t depth  = 0;
};

inline bool operator==(const Extent2D &lhs, const Extent2D &rhs) {
    return (lhs.width == rhs.width && lhs.height == rhs.height);
}

inline bool operator==(const Extent3D &lhs, const Extent3D &rhs) {
    return (lhs.width == rhs.width && lhs.height == rhs.height && lhs.depth == rhs.depth);
}

inline bool operator!=(const Extent2D &lhs, const Extent2D &rhs) {
    return !(lhs == rhs);
}

inline bool operator!=(const Extent3D &lhs, const Extent3D &rhs) {
    return !(lhs == rhs);
}

struct Offset2D {
    Offset2D() = default;

    Offset2D(const Offset2D &) = default;

    inline Offset2D(std::uint32_t x, std::uint32_t y) : x{x}, y{y} {}

    Offset2D operator+(const Offset2D &other) const {
        return {this->x + other.x, this->y + other.y};
    }

    Offset2D operator-(const Offset2D &other) const {
        return {this->x - other.x, this->y - other.y};
    }

    std::uint32_t x = 0;
    std::uint32_t y = 0;
};

struct Offset3D {
    Offset3D()                 = default;
    Offset3D(const Offset3D &) = default;

    inline Offset3D(std::uint32_t x, std::uint32_t y, std::uint32_t z) : x{x}, y{y}, z{z} {}

    Offset3D operator+(const Offset3D &other) const {
        return {this->x + other.x, this->y + other.y, this->z + other.z};
    }

    Offset3D operator-(const Offset3D &other) const {
        return {this->x - other.x, this->y - other.y, this->z + other.z};
    }

    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;
};

inline bool operator==(const Offset2D &lhs, const Offset2D &rhs) {
    return (lhs.x == rhs.x && lhs.y == rhs.y);
}

inline bool operator==(const Offset3D &lhs, const Offset3D &rhs) {
    return (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z);
}

inline bool operator!=(const Offset2D &lhs, const Offset2D &rhs) {
    return !(lhs == rhs);
}

inline bool operator!=(const Offset3D &lhs, const Offset3D &rhs) {
    return !(lhs == rhs);
}
