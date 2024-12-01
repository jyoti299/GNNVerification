import FWCore.ParameterSet.Config as cms

process = cms.Process("GNNInputs")

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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5) )

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

process.gnnInputs = cms.EDAnalyzer('GNNInputs',
    inputTracks = cms.InputTag('generalTracks'),
    SimTag = cms.InputTag('mix', 'MergedTrackTruth'),
    TPtoRecoTrackAssoc = cms.InputTag('trackingParticleRecoTrackAsssociation'),
    pathLengthSrc = cms.InputTag('trackExtenderWithMTD', 'generalTrackPathLength'),
    momentumSrc = cms.InputTag('trackExtenderWithMTD', 'generalTrackp'),
    offlineBS = cms.InputTag('offlineBeamSpot'),
    tmtd = cms.InputTag('trackExtenderWithMTD', 'generalTracktmtd'),
    timeSrc = cms.InputTag('trackExtenderWithMTD', 'generalTracktmtd'),
    sigmaSrc = cms.InputTag('trackExtenderWithMTD', 'generalTracksigmatmtd'),
    t0PID = cms.InputTag('tofPID', 't0'),
    t0SafePID = cms.InputTag('tofPID', 't0safe'),
    sigmat0SafePID = cms.InputTag('tofPID', 'sigmat0safe'),
    trackMVAQual = cms.InputTag('mtdTrackQualityMVA', 'mtdQualMVA'),
    tofPi = cms.InputTag('trackExtenderWithMTD', 'generalTrackTofPi'),
    tofK = cms.InputTag('trackExtenderWithMTD', 'generalTrackTofK'),
    tofP = cms.InputTag('trackExtenderWithMTD', 'generalTrackTofP'),
    probPi = cms.InputTag('tofPID', 'probPi'),
    probK = cms.InputTag('tofPID', 'probK'),
    probP = cms.InputTag('tofPID', 'probP'),
    sigmatofpiSrc = cms.InputTag('trackExtenderWithMTD', 'generalTrackSigmaTofPi'),
    sigmatofkSrc = cms.InputTag('trackExtenderWithMTD', 'generalTrackSigmaTofK'),
    sigmatofpSrc = cms.InputTag('trackExtenderWithMTD', 'generalTrackSigmaTofP'),
    btlMatchChi2Src = cms.InputTag('trackExtenderWithMTD', 'btlMatchChi2'),
    btlMatchTimeChi2Src = cms.InputTag('trackExtenderWithMTD', 'btlMatchTimeChi2'),
    etlMatchChi2Src = cms.InputTag('trackExtenderWithMTD', 'etlMatchChi2'),
    etlMatchTimeChi2Src = cms.InputTag('trackExtenderWithMTD', 'etlMatchTimeChi2'),
    npixBarrelSrc = cms.InputTag('trackExtenderWithMTD', 'npixBarrel'),
    npixEndcapSrc = cms.InputTag('trackExtenderWithMTD', 'npixEndcap'),
    maxD0Significance = cms.double(4),
    maxD0Error = cms.double(1),
    maxDzError = cms.double(1),
    minPt = cms.double(0),
    maxEta = cms.double(2.4),
    maxNormalizedChi2 = cms.double(10),
    minSiliconLayersWithHits = cms.int32(5),
    minPixelLayersWithHits = cms.int32(2),
)

process.p = cms.Path(process.gnnInputs)
