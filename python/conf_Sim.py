import FWCore.ParameterSet.Config as cms

process = cms.Process("SimEdgeweights")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D110Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T33', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.TFileService = cms.Service("TFileService",
    fileName = cms.string("file_check.root"),
    closeFileFast = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
                               'file:/gpfs/cms/users/jbabbar/work/VertexProd/CMSSW_14_1_0_pre6/work/TTbar_PU/246b9af9-570e-42e4-b188-7e96e65b2832.root'
                           #'/store/relval/CMSSW_14_1_0_pre5/RelValTTbar_14TeV/GEN-SIM-RECO/140X_mcRun4_realistic_v4_STD_2026D110_noPU-v1/2580000/1ad6be91-5b6a-42b7-a7a0-dad22720cd04.root',
#'/store/relval/CMSSW_14_1_0_pre5/RelValTTbar_14TeV/GEN-SIM-RECO/140X_mcRun3_2024_realistic_v11_STD_2024_noPU-v1/2580000/04803a83-4c83-47ca-8be9-ae02b17ce67a.root' 
                            )
)

process.simweights = cms.EDAnalyzer('SimEdgeweights',
    TrackTimesLabel = cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModel"),
    TrackTimeResosLabel = cms.InputTag("trackTimeValueMapProducer","generalTracksConfigurableFlatResolutionModelResolution"),    
    inputTracks = cms.InputTag('generalTracks'),
    SimTag = cms.InputTag('mix', 'MergedTrackTruth'),
    TPtoRecoTrackAssoc = cms.InputTag('trackingParticleRecoTrackAsssociation'),
    offlineBS = cms.InputTag('offlineBeamSpot'),
)

process.p = cms.Path(process.simweights)
