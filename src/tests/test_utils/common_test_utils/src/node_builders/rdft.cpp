// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/rdft.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/irdft.hpp"
#include "openvino/op/rdft.hpp"

namespace ov {
namespace test {
namespace utils {
namespace {
template <typename... Args>
std::shared_ptr<ov::Node> CallRdftCtorWithArgs(const ov::test::utils::DFTOpType op_type, Args&&... args) {
    switch (op_type) {
    case ov::test::utils::DFTOpType::FORWARD:
        return std::make_shared<ov::op::v9::RDFT>(std::forward<Args>(args)...);
    case ov::test::utils::DFTOpType::INVERSE:
        return std::make_shared<ov::op::v9::IRDFT>(std::forward<Args>(args)...);
    default:
        throw std::logic_error("Unsupported operation type");
    }
}
}  // namespace

std::shared_ptr<ov::Node> make_rdft(const ov::Output<Node>& data_node,
                                    const std::vector<int64_t>& axes,
                                    const std::vector<int64_t>& signal_size,
                                    const ov::test::utils::DFTOpType op_type) {
    auto axesNode = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{axes.size()}, axes);

    if (!signal_size.empty()) {
        auto signal_size_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64,
                                                                       ov::Shape{signal_size.size()},
                                                                       signal_size);
        return CallRdftCtorWithArgs(op_type, data_node, axesNode, signal_size_node);
    }
    return CallRdftCtorWithArgs(op_type, data_node, axesNode);
}
}  // namespace utils
}  // namespace test
}  // namespace ov
