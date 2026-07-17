/*******************************************************************************
 *
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

// Tests for the handle-based composable fused-epilogue API (RMSNorm focus).
//
// These tests validate the RMSNorm reference math plus composition rules at the API level,
// without any fused-epilogue kernels:
//  - The CPU reference implements y = x * rsqrt(mean(x^2) + eps) * gamma.
//  - The create/add/set/destroy lifecycle returns SUCCESS.
//  - An unrecognized or unsupported epilogue and an illegal/duplicate ordering are rejected
//    by hipblasLtFusedEpilogueAdd (INVALID_VALUE).
//  - Attaching an incomplete residual-add handle (unset residual pointer) is rejected at
//    descriptor-set time (INVALID_VALUE). If no residual-output pointer is provided, or if
//    it is explicitly set to NULL, the residual input is the in-place write-back target for
//    the updated residual stream.
//  - Attaching an incomplete RMSNorm handle (unset gamma or unset eps) is rejected at
//    descriptor-set time (INVALID_VALUE).
//  - The decomposed flow (partial RMSNorm stats producer + RMSNorm scale-apply consumer) is
//    linked by an opaque, library-populated RMSNorm handoff descriptor. Its create/destroy
//    lifecycle returns SUCCESS, full and decomposed RMSNorm stages cannot be mixed in one
//    chain, and attaching a decomposed handle without the handoff descriptor (or, for the
//    producer, without gamma/eps) is rejected at descriptor-set time (INVALID_VALUE).
//  - A complete-but-unimplemented fused epilogue is rejected by hipblasLtMatmul with
//    NOT_SUPPORTED before kernel selection/launch.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <random>
#include <string>
#include <vector>

namespace
{
    void cpuRmsNorm(float*       out,
                    const float* in,
                    const float* gamma,
                    std::size_t  rows,
                    std::size_t  cols,
                    float        eps)
    {
        for(std::size_t row = 0; row < rows; ++row)
        {
            const auto offset = row * cols;
            float      sum_sq = 0.0f;
            for(std::size_t col = 0; col < cols; ++col)
                sum_sq += in[offset + col] * in[offset + col];

            const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(cols) + eps);
            for(std::size_t col = 0; col < cols; ++col)
                out[offset + col] = in[offset + col] * inv_rms * gamma[col];
        }
    }

    class FusedEpilogueTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
        }
        void TearDown() override
        {
            if(fused)
                hipblasLtFusedEpilogueDestroy(fused);
            if(stats)
                hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
        }

        // Set gamma/eps so an RMSNorm handle passes attach-time validation.
        void completeRmsnorm()
        {
            int   dummy_gamma_storage = 0;
            void* gamma               = &dummy_gamma_storage;
            const float eps           = 1e-5f;
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &gamma, sizeof(gamma)),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        // Set the residual input pointer so a residual-add handle passes attach-time
        // validation. Without a residual output pointer, the API uses this pointer as the
        // in-place destination for the updated residual stream.
        void completeResidual()
        {
            int   dummy_residual_storage = 0;
            void* residual               = &dummy_residual_storage;
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &residual, sizeof(residual)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        // Create and set the opaque RMSNorm handoff descriptor so a decomposed producer or
        // consumer handle passes attach-time validation.
        void completeStats()
        {
            ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                          fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)),
                      HIPBLAS_STATUS_SUCCESS);
        }

        hipblasStatus_t attach()
        {
            // The attribute value is the handle (a pointer); pass its pointer-sized storage.
            return hipblasLtMatmulDescSetAttribute(desc,
                                                   HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                                   &fused,
                                                   sizeof(hipblasLtFusedEpilogueDescriptor_t));
        }

        hipblasLtMatmulDesc_t                     desc  = nullptr;
        hipblasLtFusedEpilogueDescriptor_t        fused = nullptr;
        hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    };
}

// ---- RMSNorm math reference ----

TEST(FusedEpilogueMath, rmsnormReferenceNormalizesRows)
{
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 4;
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> gamma(cols, 1.0f);
    std::vector<float>       output(rows * cols, 0.0f);

    cpuRmsNorm(output.data(), input.data(), gamma.data(), rows, cols, 0.0f);

    for(std::size_t row = 0; row < rows; ++row)
    {
        float mean_sq = 0.0f;
        for(std::size_t col = 0; col < cols; ++col)
        {
            const auto v = output[row * cols + col];
            mean_sq += v * v;
        }
        mean_sq /= static_cast<float>(cols);
        EXPECT_NEAR(mean_sq, 1.0f, 1e-6f);
    }
}

TEST(FusedEpilogueMath, rmsnormReferenceAppliesGamma)
{
    constexpr std::size_t rows = 1;
    constexpr std::size_t cols = 4;
    const std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> gamma = {1.0f, 0.5f, 2.0f, -1.0f};
    std::vector<float>       output(rows * cols, 0.0f);

    cpuRmsNorm(output.data(), input.data(), gamma.data(), rows, cols, 0.0f);

    const float inv_rms = 1.0f / std::sqrt(7.5f);
    EXPECT_NEAR(output[0], 1.0f * inv_rms, 1e-6f);
    EXPECT_NEAR(output[1], 2.0f * inv_rms * 0.5f, 1e-6f);
    EXPECT_NEAR(output[2], 3.0f * inv_rms * 2.0f, 1e-6f);
    EXPECT_NEAR(output[3], 4.0f * inv_rms * -1.0f, 1e-6f);
}

// ---- Lifecycle ----

TEST(FusedEpilogueLifecycle, createAddDestroy)
{
    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
    EXPECT_NE(fused, nullptr);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueDestroy(fused), HIPBLAS_STATUS_SUCCESS);
}

TEST(FusedEpilogueLifecycle, createNullRejected)
{
    EXPECT_EQ(hipblasLtFusedEpilogueCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST(FusedEpilogueLifecycle, rmsnormStatsCreateDestroy)
{
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);
    EXPECT_NE(stats, nullptr);
    EXPECT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats), HIPBLAS_STATUS_SUCCESS);
}

TEST(FusedEpilogueLifecycle, rmsnormStatsCreateNullRejected)
{
    EXPECT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(nullptr), HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- Add: ordering legality ----

TEST_F(FusedEpilogueTest, legalOrderAccepted)
{
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_AMAX),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, illegalOrderRejected)
{
    // Requant then RMSNorm violates the supported RMSNorm order (requant must come last).
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, amaxAfterRequantRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_AMAX),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, swigluRejectedByRmsnormChainValidator)
{
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_SWIGLU),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, rmsnormBeforeResidualRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, duplicateStageRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, unknownEpilogueRejected)
{
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, static_cast<hipblasLtFuseableEpilogue_t>(999)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- Add: decomposed-flow ordering and family mixing ----

TEST_F(FusedEpilogueTest, decomposedProducerOrderAccepted)
{
    // Producer chain: residual add -> partial RMSNorm stats.
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, decomposedConsumerAccepted)
{
    // Consumer chain: RMSNorm scale-apply only.
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, partialStatsBeforeResidualRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, fullRmsnormThenPartialStatsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, partialStatsThenFullRmsnormRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, producerAndConsumerStagesInOneChainRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- SetAttribute validation ----

TEST_F(FusedEpilogueTest, setUnknownAttributeRejected)
{
    const float eps = 1e-5f;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, static_cast<hipblasLtFusedEpilogueAttribute_t>(999), &eps, sizeof(eps)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setNullResidualPointerRejected)
{
    void* residual = nullptr;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &residual, sizeof(residual)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setNullRmsnormGammaRejected)
{
    void* gamma = nullptr;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &gamma, sizeof(gamma)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setNullRmsnormStatsRejected)
{
    hipblasLtFusedEpilogueRMSNormDescriptor_t null_stats = nullptr;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &null_stats, sizeof(null_stats)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setNullResidualOutputAcceptedAsInPlace)
{
    void* residual_output = nullptr;
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &residual_output,
                                                 sizeof(residual_output)),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, setInvalidRequantComputeModeRejected)
{
    auto mode = static_cast<hipblasLtRequantScaleComputeMode_t>(999);
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_COMPUTE_MODE, &mode, sizeof(mode)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, setInvalidRequantGranularityRejected)
{
    auto granularity = static_cast<hipblasLtRequantScaleGranularity_t>(999);
    EXPECT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY,
                                                 &granularity,
                                                 sizeof(granularity)),
              HIPBLAS_STATUS_INVALID_VALUE);
}

// ---- Attach-time completeness validation ----

TEST_F(FusedEpilogueTest, attachResidualMissingPointerRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    // residual pointer never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachResidualInPlaceWritebackAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    // No residual-output pointer is required; unset means update the residual input in place.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachResidualSeparateWritebackAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    int   dummy_residual_output_storage = 0;
    void* residual_output               = &dummy_residual_output_storage;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &residual_output,
                                                 sizeof(residual_output)),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachResidualOutputCanBeClearedToInPlace)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    int   dummy_residual_output_storage = 0;
    void* residual_output               = &dummy_residual_output_storage;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &residual_output,
                                                 sizeof(residual_output)),
              HIPBLAS_STATUS_SUCCESS);
    residual_output = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(fused,
                                                 HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                                 &residual_output,
                                                 sizeof(residual_output)),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachRmsnormMissingGammaRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    const float eps = 1e-5f;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
              HIPBLAS_STATUS_SUCCESS);
    // gamma never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachRmsnormMissingEpsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    int   dummy = 0;
    void* gamma = &dummy;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &gamma, sizeof(gamma)),
              HIPBLAS_STATUS_SUCCESS);
    // eps never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteRmsnormAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    completeRmsnorm();
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachResidualRmsnormAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    completeRmsnorm();
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

// ---- Attach-time completeness validation: decomposed flow ----

TEST_F(FusedEpilogueTest, attachPartialStatsMissingStatsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    completeRmsnorm();
    // stats handoff descriptor never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachPartialStatsMissingGammaEpsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    completeStats();
    // gamma/eps never set -> attach must reject (the producer computes the partial stats).
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteProducerAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    completeResidual();
    completeRmsnorm();
    completeStats();
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachScaleApplyMissingStatsRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
    // stats handoff descriptor never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteConsumerAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
    completeStats();
    // scale-apply only needs the handoff descriptor; gamma/eps live on the producer.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachNullFusedEpilogueDetaches)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    completeRmsnorm();
    ASSERT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);

    hipblasLtFusedEpilogueDescriptor_t null_fused = nullptr;
    EXPECT_EQ(hipblasLtMatmulDescSetAttribute(desc,
                                              HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                              &null_fused,
                                              sizeof(null_fused)),
              HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachRequantMissingScaleRejected)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    // scale pointer never set -> attach must reject.
    EXPECT_EQ(attach(), HIPBLAS_STATUS_INVALID_VALUE);
}

TEST_F(FusedEpilogueTest, attachCompleteRequantAccepted)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT),
              HIPBLAS_STATUS_SUCCESS);
    int   dummy_scale_storage = 0;
    void* scale               = &dummy_scale_storage;
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_POINTER, &scale, sizeof(scale)),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);
}

TEST_F(FusedEpilogueTest, attachedDecomposedConsumerMatmulNotSupported)
{
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
    completeStats();
    ASSERT_EQ(attach(), HIPBLAS_STATUS_SUCCESS);

    EXPECT_EQ(hipblasLtMatmul(nullptr,
                              desc,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              0,
                              nullptr),
              HIPBLAS_STATUS_NOT_SUPPORTED);
}

// ---- End-to-end numeric test: full RMSNorm flow on device ----
//
// Drives a real bf16 TN matmul through hipblasLtMatmul with a full-RMSNorm fused epilogue
// attached, then compares D against a CPU reference RMSNorm(alpha * op(A)*op(B), gamma, eps).
// This exercises the wired path end to end: solution selection (UsePartialRMS predicate),
// K1 (GEMM + partial-stats producer), and Kernel 2 (reduce-and-apply). gfx950-only, since the
// PartialRMS solution + partial_rms_epilogue code object ship for gfx950.

namespace
{
    inline uint16_t f32_to_bf16(float f)
    {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        // Round to nearest even.
        const uint32_t lsb = (bits >> 16) & 1u;
        bits += 0x7fffu + lsb;
        return static_cast<uint16_t>(bits >> 16);
    }

    inline float bf16_to_f32(uint16_t h)
    {
        const uint32_t bits = static_cast<uint32_t>(h) << 16;
        float          f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    bool deviceIsGfx950()
    {
        int dev = 0;
        if(hipGetDevice(&dev) != hipSuccess)
            return false;
        hipDeviceProp_t prop{};
        if(hipGetDeviceProperties(&prop, dev) != hipSuccess)
            return false;
        return std::string(prop.gcnArchName).rfind("gfx950", 0) == 0;
    }
}

TEST(FusedEpilogueE2E, fullRmsNormMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "fused RMSNorm (PartialRMS) is wired for gfx950 only";

    // TN, bf16, col-major. Shape matches a gfx950 PartialRMS logic entry (M=N=1024, K=4096).
    const int64_t M = 1024, N = 1024, K = 4096;
    const float   eps = 1e-5f, alpha = 1.0f, beta = 0.0f;

    // Host inputs. op(A)=T => A stored K x M col-major; op(B)=N => B stored K x N col-major.
    std::vector<uint16_t> hA(static_cast<size_t>(K) * M);
    std::vector<uint16_t> hB(static_cast<size_t>(K) * N);
    std::vector<uint16_t> hGamma(N);
    std::vector<uint16_t> hD(static_cast<size_t>(M) * N, 0);

    std::mt19937                          rng(123);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> gdist(0.5f, 1.5f);
    for(auto& x : hA)
        x = f32_to_bf16(dist(rng));
    for(auto& x : hB)
        x = f32_to_bf16(dist(rng));
    for(auto& x : hGamma)
        x = f32_to_bf16(gdist(rng));

    // Device buffers.
    void*        dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dGamma = nullptr,
        *dWs                 = nullptr;
    const size_t wsSize      = size_t(256) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, hD.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    dC = dD; // beta = 0, C unused numerically but must be a valid pointer.

    ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, hD.size() * sizeof(uint16_t)), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr, layD = nullptr;
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layA, HIP_R_16BF, K, M, K), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layB, HIP_R_16BF, K, N, K), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layC, HIP_R_16BF, M, N, M), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layD, HIP_R_16BF, M, N, M), HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatmulDesc_t mm = nullptr;
    ASSERT_EQ(hipblasLtMatmulDescCreate(&mm, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              HIPBLAS_STATUS_SUCCESS);
    const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)),
              HIPBLAS_STATUS_SUCCESS);

    // Attach a full-RMSNorm fused epilogue with device gamma + eps.
    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &dGamma, sizeof(dGamma)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(mm,
                                              HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                              &fused,
                                              sizeof(hipblasLtFusedEpilogueDescriptor_t)),
              HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                  pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)),
              HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatmulHeuristicResult_t heur[1];
    int                              algoCount = 0;
    ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                  handle, mm, layA, layB, layC, layD, pref, 1, heur, &algoCount),
              HIPBLAS_STATUS_SUCCESS);
    // The UsePartialRMS predicate must route this fused problem to the PartialRMS solution.
    ASSERT_GT(algoCount, 0) << "no PartialRMS solution selected for the fused RMSNorm problem";

    ASSERT_EQ(hipblasLtMatmul(handle,
                              mm,
                              &alpha,
                              dA,
                              layA,
                              dB,
                              layB,
                              &beta,
                              dC,
                              layC,
                              dD,
                              layD,
                              &heur[0].algo,
                              dWs,
                              wsSize,
                              nullptr),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
              hipSuccess);

    // CPU reference: h1[m,n] = alpha * sum_k A[k,m] * B[k,n]; then RMSNorm over n with gamma.
    std::vector<float> gammaF(N);
    for(int64_t j = 0; j < N; ++j)
        gammaF[j] = bf16_to_f32(hGamma[j]);

    size_t mismatches   = 0;
    double maxRelErr    = 0.0;
    for(int64_t m = 0; m < M; ++m)
    {
        std::vector<float> h1(N);
        float              sumSq = 0.0f;
        for(int64_t n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for(int64_t kk = 0; kk < K; ++kk)
                acc += bf16_to_f32(hA[kk + m * K]) * bf16_to_f32(hB[kk + n * K]);
            acc *= alpha;
            h1[n] = acc;
            sumSq += acc * acc;
        }
        const float invRms = 1.0f / std::sqrt(sumSq / static_cast<float>(N) + eps);
        for(int64_t n = 0; n < N; ++n)
        {
            const float ref = h1[n] * invRms * gammaF[n];
            // Fused RMSNorm returns D row-major [M, N_hidden] (N_hidden contiguous):
            // the PartialRMS K1 emitter reduces free0 = N_hidden, so the host transposes
            // the GEMM (free0 = N_hidden) which lands D as row-major.
            const float got = bf16_to_f32(hD[m * N + n]); // D row-major [M, N]
            const float denom = std::max(std::abs(ref), 1e-3f);
            const double rel   = std::abs(got - ref) / denom;
            maxRelErr          = std::max(maxRelErr, static_cast<double>(rel));
            if(rel > 5e-2)
                ++mismatches;
        }
    }

    EXPECT_EQ(mismatches, 0u) << "max relative error " << maxRelErr;

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtFusedEpilogueDestroy(fused);
    hipblasLtMatmulDescDestroy(mm);
    hipblasLtMatrixLayoutDestroy(layA);
    hipblasLtMatrixLayoutDestroy(layB);
    hipblasLtMatrixLayoutDestroy(layC);
    hipblasLtMatrixLayoutDestroy(layD);
    hipblasLtDestroy(handle);
    hipFree(dA);
    hipFree(dB);
    hipFree(dD);
    hipFree(dGamma);
    hipFree(dWs);
}

// ---- End-to-end numeric test: full RMSNorm flow with a fused residual-add stage ----
//
// Same as fullRmsNormMatchesReference, but chains HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD
// before the RMSNorm stage. K1 (PartialRMS producer) adds the residual to the GEMM output
// (H = alpha * op(A)*op(B) + residual) before the sum-of-squares and gamma, and Kernel 2
// normalizes H in place. The residual is a bf16 row-major [M, N] tensor (offset m*N + n),
// matching the kernel's ResidualBuf addressing and the TensileLite client reference. This
// exercises the residualAdd=True PartialRMS solution + predicate. gfx950-only.

TEST(FusedEpilogueE2E, fullRmsNormResidualAddMatchesReference)
{
    // Skipped until the residual-add PartialRMS K1 solution passes validation and ships in
    // the device library. The residual path is fully wired on the host/runtime side, and the
    // non-residual PartialRMS path validates end to end (see fullRmsNormMatchesReference).
    //
    // Provisioning mechanism (post epilogues/ reorg): the residualAdd=True K1 solution is
    // generated from the row-major benchmark YAML, which forks both PartialRMSResidualAdd
    // False/True groups, via the one-shot build script:
    //   epilogues/scripts/build_library.sh --yaml epilogues/yaml/gemm_partial_rms_k1_rowmajor.yaml
    // (the old epilogues/gen/gen_partialrms_logic.py generator was removed in that reorg).
    // At runtime the residual vs non-residual variants are disambiguated by the
    // UsePartialRMSResidualAddEqual predicate (ContractionProblemPredicates.hpp).
    //
    // Blocker: the residualAdd=True K1 kernel currently FAILS partialBuf validation in the
    // Tensile benchmark -- the device per-row partial sum-of-squares is ~6.24x the reference
    // for every element (e.g. 2.99412e6 vs 479419) on 4096x4096 and 8192x4096. Because the
    // solution does not validate, it is not built into the shipped library, so this test
    // stays skipped. Re-enable once the residualAdd partialBuf correctness issue (kernel
    // SubtilePartialRMSEmit residual path vs the client/CPU reference) is resolved. The body
    // below is complete and validates residual-add end to end once the solution ships.
    if(!deviceIsGfx950())
        GTEST_SKIP() << "fused RMSNorm (PartialRMS) is wired for gfx950 only";

    // TN, bf16, col-major. Shape matches a gfx950 PartialRMS logic entry (M=N=1024, K=4096).
    const int64_t M = 1024, N = 1024, K = 4096;
    const float   eps = 1e-5f, alpha = 1.0f, beta = 0.0f;

    // Host inputs. op(A)=T => A stored K x M col-major; op(B)=N => B stored K x N col-major.
    std::vector<uint16_t> hA(static_cast<size_t>(K) * M);
    std::vector<uint16_t> hB(static_cast<size_t>(K) * N);
    std::vector<uint16_t> hGamma(N);
    // Residual is column-major [M, N] (same layout as D): residual(m, n) at offset m + n*M.
    std::vector<uint16_t> hResidual(static_cast<size_t>(M) * N);
    std::vector<uint16_t> hD(static_cast<size_t>(M) * N, 0);

    std::mt19937                          rng(123);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> gdist(0.5f, 1.5f);
    for(auto& x : hA)
        x = f32_to_bf16(dist(rng));
    for(auto& x : hB)
        x = f32_to_bf16(dist(rng));
    for(auto& x : hGamma)
        x = f32_to_bf16(gdist(rng));
    for(auto& x : hResidual)
        x = f32_to_bf16(dist(rng));

    // Device buffers.
    void*        dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dGamma = nullptr,
        *dResidual = nullptr, *dWs = nullptr;
    const size_t wsSize      = size_t(256) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, hD.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dResidual, hResidual.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    dC = dD; // beta = 0, C unused numerically but must be a valid pointer.

    ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dGamma, hGamma.data(), hGamma.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dResidual, hResidual.data(), hResidual.size() * sizeof(uint16_t),
                        hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, hD.size() * sizeof(uint16_t)), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr, layD = nullptr;
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layA, HIP_R_16BF, K, M, K), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layB, HIP_R_16BF, K, N, K), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layC, HIP_R_16BF, M, N, M), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layD, HIP_R_16BF, M, N, M), HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatmulDesc_t mm = nullptr;
    ASSERT_EQ(hipblasLtMatmulDescCreate(&mm, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              HIPBLAS_STATUS_SUCCESS);
    const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)),
              HIPBLAS_STATUS_SUCCESS);

    // Attach a residual-add -> RMSNorm fused epilogue. RESIDUAL_ADD (chain rank 0) must be
    // added before the RMSNorm stage (chain rank 1).
    hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&fused), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &dGamma, sizeof(dGamma)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER, &dResidual, sizeof(dResidual)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(mm,
                                              HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                              &fused,
                                              sizeof(hipblasLtFusedEpilogueDescriptor_t)),
              HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                  pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)),
              HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatmulHeuristicResult_t heur[1];
    int                              algoCount = 0;
    ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                  handle, mm, layA, layB, layC, layD, pref, 1, heur, &algoCount),
              HIPBLAS_STATUS_SUCCESS);
    // The UsePartialRMS + UsePartialRMSResidualAdd predicates must route this to the
    // residualAdd=True PartialRMS solution.
    ASSERT_GT(algoCount, 0)
        << "no residualAdd PartialRMS solution selected for the fused RMSNorm problem";

    ASSERT_EQ(hipblasLtMatmul(handle,
                              mm,
                              &alpha,
                              dA,
                              layA,
                              dB,
                              layB,
                              &beta,
                              dC,
                              layC,
                              dD,
                              layD,
                              &heur[0].algo,
                              dWs,
                              wsSize,
                              nullptr),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
              hipSuccess);

    // CPU reference: H[m,n] = alpha * sum_k A[k,m]*B[k,n] + residual[m,n]; then RMSNorm over n.
    std::vector<float> gammaF(N);
    for(int64_t j = 0; j < N; ++j)
        gammaF[j] = bf16_to_f32(hGamma[j]);

    size_t mismatches = 0;
    double maxRelErr  = 0.0;
    for(int64_t m = 0; m < M; ++m)
    {
        std::vector<float> h1(N);
        float              sumSq = 0.0f;
        for(int64_t n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for(int64_t kk = 0; kk < K; ++kk)
                acc += bf16_to_f32(hA[kk + m * K]) * bf16_to_f32(hB[kk + n * K]);
            acc *= alpha;
            acc += bf16_to_f32(hResidual[m * N + n]); // residual row-major [M, N] (like D)
            h1[n] = acc;
            sumSq += acc * acc;
        }
        const float invRms = 1.0f / std::sqrt(sumSq / static_cast<float>(N) + eps);
        for(int64_t n = 0; n < N; ++n)
        {
            const float  ref   = h1[n] * invRms * gammaF[n];
            const float  got   = bf16_to_f32(hD[m * N + n]); // D row-major [M, N]
            const float  denom = std::max(std::abs(ref), 1e-3f);
            const double rel   = std::abs(got - ref) / denom;
            maxRelErr          = std::max(maxRelErr, static_cast<double>(rel));
            if(rel > 5e-2)
                ++mismatches;
        }
    }

    EXPECT_EQ(mismatches, 0u) << "max relative error " << maxRelErr;

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtFusedEpilogueDestroy(fused);
    hipblasLtMatmulDescDestroy(mm);
    hipblasLtMatrixLayoutDestroy(layA);
    hipblasLtMatrixLayoutDestroy(layB);
    hipblasLtMatrixLayoutDestroy(layC);
    hipblasLtMatrixLayoutDestroy(layD);
    hipblasLtDestroy(handle);
    hipFree(dA);
    hipFree(dB);
    hipFree(dD);
    hipFree(dGamma);
    hipFree(dResidual);
    hipFree(dWs);
}

// ---- End-to-end numeric test: decomposed RMSNorm consumer (Kernel 3 RstdScale) ----
//
// Exercises the decomposed flow's consumer stage in isolation: a GEMM2 with the
// HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY epilogue multiplies each output row by a
// pre-computed per-row rstd carried in the handoff descriptor (K3 RstdScale, normal
// orientation, no reduction). The decomposed producer reduce-and-return kernel is not yet
// implemented, so the test injects a host-computed rstd into the opaque handoff via a
// test-only hook. Verifies D[m,n] = (alpha * op(A)*op(B))[m,n] * rstd[m]. gfx950-only.

// Test-only hook (defined in amd_detail/hipblaslt.cpp) to populate the opaque RMSNorm handoff
// descriptor with a caller-provided device rstd buffer before the producer kernel exists.
extern "C++" bool rocblaslt_rmsnorm_handoff_set_scale_for_testing(
    hipblasLtFusedEpilogueRMSNormDescriptor_t desc, void* per_row_scale);

TEST(FusedEpilogueE2E, decomposedScaleApplyMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "fused RMSNorm (RstdScale) is wired for gfx950 only";

    // TN, bf16, col-major. K3 RstdScale library tiles are N_out=64 wide; K = N_hidden.
    const int64_t M = 256, N = 64, K = 64;
    const float   alpha = 1.0f, beta = 0.0f;

    std::vector<uint16_t> hA(static_cast<size_t>(K) * M);
    std::vector<uint16_t> hB(static_cast<size_t>(K) * N);
    std::vector<uint16_t> hD(static_cast<size_t>(M) * N, 0);
    std::vector<float>    hRstd(static_cast<size_t>(M));

    std::mt19937                          rng(321);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> rdist(0.25f, 1.75f);
    for(auto& x : hA)
        x = f32_to_bf16(dist(rng));
    for(auto& x : hB)
        x = f32_to_bf16(dist(rng));
    for(auto& r : hRstd)
        r = rdist(rng); // arbitrary per-row scale standing in for the producer's rstd

    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dRstd = nullptr,
         *dWs                = nullptr;
    const size_t wsSize      = size_t(64) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dA, hA.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dB, hB.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD, hD.size() * sizeof(uint16_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dRstd, hRstd.size() * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    dC = dD;
    ASSERT_EQ(hipMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dB, hB.data(), hB.size() * sizeof(uint16_t), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dRstd, hRstd.data(), hRstd.size() * sizeof(float), hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemset(dD, 0, hD.size() * sizeof(uint16_t)), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr, layD = nullptr;
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layA, HIP_R_16BF, K, M, K), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layB, HIP_R_16BF, K, N, K), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layC, HIP_R_16BF, M, N, M), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layD, HIP_R_16BF, M, N, M), HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatmulDesc_t mm = nullptr;
    ASSERT_EQ(hipblasLtMatmulDescCreate(&mm, HIPBLAS_COMPUTE_32F, HIP_R_32F),
              HIPBLAS_STATUS_SUCCESS);
    const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)),
              HIPBLAS_STATUS_SUCCESS);

    // Decomposed handoff, populated with the host rstd via the test-only hook.
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(rocblaslt_rmsnorm_handoff_set_scale_for_testing(stats, dRstd));

    // Consumer chain: RMSNorm scale-apply reads the deferred per-row scale from the handoff.
    hipblasLtFusedEpilogueDescriptor_t cons = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&cons), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(cons, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  cons, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                  mm, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &cons, sizeof(cons)),
              HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatmulPreference_t pref = nullptr;
    ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                  pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize)),
              HIPBLAS_STATUS_SUCCESS);

    hipblasLtMatmulHeuristicResult_t heur[1];
    int                              algoCount = 0;
    ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                  handle, mm, layA, layB, layC, layD, pref, 1, heur, &algoCount),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no RstdScale (K3) solution selected for the scale-apply problem";

    ASSERT_EQ(hipblasLtMatmul(handle, mm, &alpha, dA, layA, dB, layB, &beta, dC, layC, dD, layD,
                              &heur[0].algo, dWs, wsSize, nullptr),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_EQ(hipMemcpy(hD.data(), dD, hD.size() * sizeof(uint16_t), hipMemcpyDeviceToHost),
              hipSuccess);

    // Reference: D[m,n] = (alpha * sum_k A[k,m]*B[k,n]) * rstd[m]. K3 keeps the normal
    // orientation (per-M-row scale, no reduction), so D is col-major [M, N].
    size_t mismatches = 0;
    double maxRelErr  = 0.0;
    for(int64_t m = 0; m < M; ++m)
        for(int64_t n = 0; n < N; ++n)
        {
            float acc = 0.0f;
            for(int64_t kk = 0; kk < K; ++kk)
                acc += bf16_to_f32(hA[kk + m * K]) * bf16_to_f32(hB[kk + n * K]);
            const float  ref   = acc * alpha * hRstd[m];
            const float  got   = bf16_to_f32(hD[n * M + m]); // D col-major [M, N]
            const float  denom = std::max(std::abs(ref), 1e-3f);
            const double rel   = std::abs(got - ref) / denom;
            maxRelErr          = std::max(maxRelErr, static_cast<double>(rel));
            if(rel > 5e-2)
                ++mismatches;
        }
    EXPECT_EQ(mismatches, 0u) << "max relative error " << maxRelErr;

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtFusedEpilogueDestroy(cons);
    hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
    hipblasLtMatmulDescDestroy(mm);
    hipblasLtMatrixLayoutDestroy(layA);
    hipblasLtMatrixLayoutDestroy(layB);
    hipblasLtMatrixLayoutDestroy(layC);
    hipblasLtMatrixLayoutDestroy(layD);
    hipblasLtDestroy(handle);
    hipFree(dA);
    hipFree(dB);
    hipFree(dD);
    hipFree(dRstd);
    hipFree(dWs);
}

// ---- End-to-end: decomposed producer(GEMM1) -> consumer(GEMM2) two-call flow ----
//
// The full decomposed RMSNorm flow across two matmul calls linked by the library-populated
// RMSNorm handoff descriptor (no test hook):
//   GEMM1 (producer, PARTIAL_RMSNORM_STATS): h2 = (x @ W0) * gamma  [M, N_hidden]; the library
//     runs K1 (PartialRMS) + row_rstd, stashing rstd = rsqrt(mean(h1^2)+eps) in the handoff.
//   GEMM2 (consumer, RMSNORM_SCALE_APPLY):    y  = rstd * (h2 @ W1)  [M, N_out] via Kernel 3.
// The combined result equals RMSNorm(x @ W0) @ W1. gamma=1 keeps the reference simple. TN bf16;
// h2 is produced row-major [M, N_hidden] and fed to GEMM2 as its TN A operand (lda=N_hidden).
// Needs a merged K1(PartialRMS)+K3(RstdScale) gfx950 library; gfx950-only.
namespace
{
    // TN (opA=T, opB=N) bf16 GEMM D[m,n]=op(A)op(B) with a fused epilogue attached. A is [k,m]
    // col-major (ld=lda), B is [k,n] col-major (ld=k), C/D are [m,n] col-major (ld=m).
    hipblasStatus_t runTnBf16GemmFused(hipblasLtHandle_t                   handle,
                                       int64_t                            m,
                                       int64_t                            n,
                                       int64_t                            k,
                                       void*                              dA,
                                       int64_t                            lda,
                                       void*                              dB,
                                       void*                              dC,
                                       void*                              dD,
                                       hipblasLtFusedEpilogueDescriptor_t fused,
                                       void*                              dWs,
                                       size_t                             wsSize,
                                       int*                               algoCount)
    {
        *algoCount = 0;
        hipblasLtMatrixLayout_t lA = nullptr, lB = nullptr, lC = nullptr, lD = nullptr;
        hipblasLtMatrixLayoutCreate(&lA, HIP_R_16BF, k, m, lda);
        hipblasLtMatrixLayoutCreate(&lB, HIP_R_16BF, k, n, k);
        hipblasLtMatrixLayoutCreate(&lC, HIP_R_16BF, m, n, m);
        hipblasLtMatrixLayoutCreate(&lD, HIP_R_16BF, m, n, m);
        hipblasLtMatmulDesc_t mm = nullptr;
        hipblasLtMatmulDescCreate(&mm, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        hipblasLtMatmulDescSetAttribute(mm, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        hipblasLtMatmulDescSetAttribute(
            mm, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, &fused, sizeof(fused));
        hipblasLtMatmulPreference_t pref = nullptr;
        hipblasLtMatmulPreferenceCreate(&pref);
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsSize, sizeof(wsSize));
        hipblasLtMatmulHeuristicResult_t heur[1];
        hipblasLtMatmulAlgoGetHeuristic(handle, mm, lA, lB, lC, lD, pref, 1, heur, algoCount);
        float           alpha = 1.0f, beta = 0.0f;
        hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
        if(*algoCount > 0)
            status = hipblasLtMatmul(handle, mm, &alpha, dA, lA, dB, lB, &beta, dC, lC, dD, lD,
                                     &heur[0].algo, dWs, wsSize, nullptr);
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(mm);
        hipblasLtMatrixLayoutDestroy(lA);
        hipblasLtMatrixLayoutDestroy(lB);
        hipblasLtMatrixLayoutDestroy(lC);
        hipblasLtMatrixLayoutDestroy(lD);
        return status;
    }
}

TEST(FusedEpilogueE2E, decomposedProducerConsumerMatchesReference)
{
    if(!deviceIsGfx950())
        GTEST_SKIP() << "decomposed RMSNorm flow is wired for gfx950 only";

    const int64_t M = 1024, Nhidden = 1024, K0 = 64, Nout = 64;
    const float   eps = 1e-5f;

    std::vector<uint16_t> hX(static_cast<size_t>(K0) * M);
    std::vector<uint16_t> hW0(static_cast<size_t>(K0) * Nhidden);
    std::vector<uint16_t> hW1(static_cast<size_t>(Nhidden) * Nout);
    std::vector<uint16_t> hGamma(static_cast<size_t>(Nhidden), f32_to_bf16(1.0f)); // gamma = 1

    std::mt19937                          rng(4242);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for(auto& v : hX)
        v = f32_to_bf16(dist(rng));
    for(auto& v : hW0)
        v = f32_to_bf16(dist(rng));
    for(auto& v : hW1)
        v = f32_to_bf16(dist(rng));

    void *dX = nullptr, *dW0 = nullptr, *dGamma = nullptr, *dH2 = nullptr, *dW1 = nullptr,
         *dD2                = nullptr, *dWs = nullptr;
    const size_t wsSize      = size_t(256) * 1024 * 1024;
    ASSERT_EQ(hipMalloc(&dX, hX.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW0, hW0.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dGamma, hGamma.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dH2, size_t(M) * Nhidden * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dW1, hW1.size() * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dD2, size_t(M) * Nout * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&dWs, wsSize), hipSuccess);
    ASSERT_EQ(hipMemcpy(dX, hX.data(), hX.size() * 2, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dW0, hW0.data(), hW0.size() * 2, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(dGamma, hGamma.data(), hGamma.size() * 2, hipMemcpyHostToDevice),
              hipSuccess);
    ASSERT_EQ(hipMemcpy(dW1, hW1.data(), hW1.size() * 2, hipMemcpyHostToDevice), hipSuccess);

    hipblasLtHandle_t handle = nullptr;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    // Library-populated handoff shared by both calls (no test hook).
    hipblasLtFusedEpilogueRMSNormDescriptor_t stats = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats), HIPBLAS_STATUS_SUCCESS);

    // Producer chain: partial RMSNorm stats + gamma + eps + handoff.
    hipblasLtFusedEpilogueDescriptor_t prod = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&prod), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(prod, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  prod, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA, &dGamma, sizeof(dGamma)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  prod, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS, &eps, sizeof(eps)),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  prod, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)),
              HIPBLAS_STATUS_SUCCESS);

    // GEMM1 producer: h2 [M, N_hidden] (row-major) + rstd stashed in the handoff.
    int algoCount = 0;
    ASSERT_EQ(runTnBf16GemmFused(handle, M, Nhidden, K0, dX, K0, dW0, dH2, dH2, prod, dWs, wsSize,
                                 &algoCount),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no PartialRMS (K1) producer solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Consumer chain: scale-apply + the same handoff.
    hipblasLtFusedEpilogueDescriptor_t cons = nullptr;
    ASSERT_EQ(hipblasLtFusedEpilogueCreate(&cons), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueAdd(cons, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtFusedEpilogueSetAttribute(
                  cons, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS, &stats, sizeof(stats)),
              HIPBLAS_STATUS_SUCCESS);

    // GEMM2 consumer: h2 (row-major [M, N_hidden]) is the TN A operand [N_hidden, M] (lda=N_hidden).
    ASSERT_EQ(runTnBf16GemmFused(handle, M, Nout, Nhidden, dH2, Nhidden, dW1, dD2, dD2, cons, dWs,
                                 wsSize, &algoCount),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_GT(algoCount, 0) << "no RstdScale (K3) consumer solution selected";
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    std::vector<uint16_t> hD2(static_cast<size_t>(M) * Nout);
    ASSERT_EQ(hipMemcpy(hD2.data(), dD2, hD2.size() * 2, hipMemcpyDeviceToHost), hipSuccess);

    // Reference: gemm1 = x@W0 (TN); rstd = rsqrt(mean(gemm1^2)+eps); y = rstd * (gemm1 @ W1).
    std::vector<float> gemm1(static_cast<size_t>(M) * Nhidden);
    std::vector<float> rstd(static_cast<size_t>(M));
    for(int64_t m = 0; m < M; ++m)
    {
        float ss = 0.0f;
        for(int64_t j = 0; j < Nhidden; ++j)
        {
            float acc = 0.0f;
            for(int64_t k = 0; k < K0; ++k)
                acc += bf16_to_f32(hX[k + m * K0]) * bf16_to_f32(hW0[k + j * K0]);
            gemm1[m * Nhidden + j] = acc;
            ss += acc * acc;
        }
        rstd[m] = 1.0f / std::sqrt(ss / static_cast<float>(Nhidden) + eps);
    }
    // Consumer reference. Mirror the device's bf16 storage: h2 = bf16(gemm1) (gamma=1) is stored
    // by the producer, GEMM2 reads it, then y = bf16(rstd * (h2 @ W1)). Compare with a combined
    // absolute+relative tolerance so near-zero cancellation elements (tiny ref) do not blow up a
    // pure relative metric.
    size_t mismatches = 0;
    double maxAbsErr  = 0.0;
    for(int64_t m = 0; m < M; ++m)
        for(int64_t n = 0; n < Nout; ++n)
        {
            float acc = 0.0f;
            for(int64_t j = 0; j < Nhidden; ++j)
            {
                const float h2bf = bf16_to_f32(f32_to_bf16(gemm1[m * Nhidden + j])); // gamma=1
                acc += h2bf * bf16_to_f32(hW1[j + n * Nhidden]);
            }
            const float ref    = bf16_to_f32(f32_to_bf16(acc * rstd[m]));
            const float got    = bf16_to_f32(hD2[n * M + m]); // D2 col-major [M, N_out]
            const float absErr = std::abs(got - ref);
            maxAbsErr          = std::max(maxAbsErr, static_cast<double>(absErr));
            if(absErr > std::max(3e-2f, 5e-2f * std::abs(ref)))
                ++mismatches;
        }
    EXPECT_EQ(mismatches, 0u) << "max abs error " << maxAbsErr;

    hipblasLtFusedEpilogueDestroy(prod);
    hipblasLtFusedEpilogueDestroy(cons);
    hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
    hipblasLtDestroy(handle);
    hipFree(dX);
    hipFree(dW0);
    hipFree(dGamma);
    hipFree(dH2);
    hipFree(dW1);
    hipFree(dD2);
    hipFree(dWs);
}
