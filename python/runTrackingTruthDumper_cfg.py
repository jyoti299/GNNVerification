import FWCore.ParameterSet.Config as cms

process = cms.Process("tPtVDump")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/gfsvol01/cms/users/jbabbar/work/container/Test_rel/CMSSW_14_1_0_pre6/work/revtx_step3PU.root')
    #/gpfs/cms/users/jbabbar/work/VertexProd/CMSSW_14_1_0_pre6/work/TTbar_PU/246b9af9-570e-42e4-b188-7e96e65b2832.root')
)

process.load("SimGeneral.TrackingAnalysis.trackingTruthDumper_cfi")
process.trackingTruthDumper.dumpVtx = True
process.trackingTruthDumper.dumpTk = True

process.p1 = cms.Path(process.trackingTruthDumper)
