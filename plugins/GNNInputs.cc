// -*- C++ -*-
//
// Package:    GNN_Ntuples/GNNInputs
// Class:      GNNInputs
//
/**\class GNNInputs GNNInputs.cc GNN_Ntuples/GNNInputs/plugins/GNNInputs.cc

 Description: Store the track variables for GNN Vertexing training 

*/
//
// Original Author:  Ksenia De Leo
//         Created:  Wed, 24 Jul 2024 14:32:53 GMT
//
//

// system include files
#include <memory>

// user include files
#include "TTree.h"
#include "TFile.h"
#include <TH2D.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

// TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

// reco track and vertex
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

//
// class declaration
//

using reco::TrackCollection;

class GNNInputs : public edm::one::EDAnalyzer<edm::one::SharedResources> {
	typedef math::XYZTLorentzVector LorentzVector;

   struct simPrimaryVertex {
    simPrimaryVertex(double x1, double y1, double z1, double t1)
        : x(x1),
          y(y1),
          z(z1),
          t(t1),
          pt(0),
          ptsq(0),
          closest_vertex_distance_z(-1.),
          nGenTrk(0),
          num_matched_reco_tracks(0),
          average_match_quality(0.0) {
      ptot.setPx(0);
      ptot.setPy(0);
      ptot.setPz(0);
      ptot.setE(0);
      p4 = LorentzVector(0, 0, 0, 0);
      r = sqrt(x * x + y * y);
    };
    double x, y, z, r, t;
    HepMC::FourVector ptot;
    LorentzVector p4;
    double pt;
    double ptsq;
    double closest_vertex_distance_z;
    int nGenTrk;
    int num_matched_reco_tracks;
    float average_match_quality;
    EncodedEventId eventId;
    TrackingVertexRef sim_vertex;
    int OriginalIndex = -1;

    unsigned int nwosmatch = 0;                    // number of rec vertices dominated by this sim evt (by wos)
    unsigned int nwntmatch = 0;                    // number of rec vertices dominated by this sim evt  (by wnt)
    std::vector<unsigned int> wos_dominated_recv;  // list of dominated rec vertices (by wos, size==nwosmatch)

    std::map<unsigned int, double> wnt;  // weighted number of tracks in recvtx (by index)
    std::map<unsigned int, double> wos;  // weight over sigma**2 in recvtx (by index)
    double sumwos = 0;                   // sum of wos in any recvtx
    double sumwnt = 0;                   // sum of weighted tracks
    unsigned int matchQuality = 0;       // quality flag
};

  // auxiliary class holding reconstructed vertices
  struct recoPrimaryVertex {
    recoPrimaryVertex(double x1, double y1, double z1)
        : x(x1),
          y(y1),
          z(z1),
          pt(0),
          ptsq(0),
          closest_vertex_distance_z(-1.),
          nRecoTrk(0),
          num_matched_sim_tracks(0),
          ndof(0.),
          recVtx(nullptr) {
      r = sqrt(x * x + y * y);
    };
    double x, y, z, r;
    double pt;
    double ptsq;
    double closest_vertex_distance_z;
    int nRecoTrk;
    int num_matched_sim_tracks;
    double ndof;
    const reco::Vertex* recVtx;
    reco::VertexBaseRef recVtxRef;
    int OriginalIndex = -1;
  };    
public:
  explicit GNNInputs(const edm::ParameterSet&);
  ~GNNInputs() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  const edm::Ref<std::vector<TrackingParticle>>* getAnyMatchedTP(const reco::TrackBaseRef&);
  std::pair<const edm::Ref<std::vector<TrackingParticle>>*, int> getMatchedTP(const reco::TrackBaseRef&,
                                                                              const TrackingVertexRef&);
  std::vector<GNNInputs::simPrimaryVertex> getSimPVs(const edm::Handle<TrackingVertexCollection>&);
  std::vector<GNNInputs::recoPrimaryVertex> getRecoPVs(const edm::Handle<edm::View<reco::Vertex>>&);
  double timeFromTrueMass(double, double, double, double);
  bool passTrackFilter(const reco::TransientTrack&);

  edm::Service<TFileService> fs_;
  std::map<unsigned int, TTree*> eventTrees_;
  //std::vector<std::unique_ptr<TH2D>> histograms;
  // ----------member data ---------------------------
  edm::EDGetTokenT<TrackCollection> tracksToken_;  //used to select what tracks to read from configuration file
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> theTTBToken;
  edm::EDGetTokenT<edm::View<reco::Vertex>> Rec4DVerToken_;
  static constexpr double simUnit_ = 1e9;     //sim time in s while reco time in ns
  static constexpr double c_ = 2.99792458e1;  //c in cm/ns
  std::string histName_;
  const reco::RecoToSimCollection* r2s_;

  float maxD0Sig_;
  float maxD0Error_;
  float maxDzError_;
  float minPt_;
  float maxEta_;
  float maxNormChi2_;
  int minSiLayers_;
  int minPxLayers_;
  const double trackweightTh_;
  // gnn input variables
  double gnn_pt;
  double gnn_eta;
  double gnn_phi;
  double gnn_z_pca;
  double gnn_dz;
  double gnn_t_pi;
  double gnn_t_k;
  double gnn_t_p;
  double gnn_sigma_t0safe;
  double gnn_sigma_tmtd;
  double gnn_t0safe;
  double gnn_t0pid;
  double gnn_mva_qual;
  double gnn_btlMatchChi2;
  double gnn_btlMatchTimeChi2;
  double gnn_etlMatchChi2;
  double gnn_etlMatchTimeChi2;
  double gnn_pathLength;
  int gnn_npixBarrel;
  int gnn_npixEndcap;
  double gnn_mtdTime;
  int gnn_is_matched_tp;
  int gnn_sim_vertex_evID;
  int gnn_sim_vertex_BX;
  double gnn_sim_vertex_z;
  double gnn_sim_vertex_t;
  int gnn_sim_vertex_index;
  double gnn_tp_tEst;
  int gnn_tp_pdgId;
  double gnn_probPi;
  double gnn_probK;
  double gnn_probP;
  double gnn_trk_chi2;
  double gnn_trk_ndof;
  int gnn_trk_validhits;
  double gnn_sigma_tof_Pi;
  double gnn_sigma_tof_K;
  double gnn_sigma_tof_P;

  edm::EDGetTokenT<edm::ValueMap<float>> btlMatchChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float>> btlMatchTimeChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float>> etlMatchChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float>> etlMatchTimeChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<int>> npixBarrelToken_;
  edm::EDGetTokenT<edm::ValueMap<int>> npixEndcapToken_;

  edm::EDGetTokenT<reco::TrackCollection> RecTrackToken_;
  edm::EDGetTokenT<reco::BeamSpot> RecBeamSpotToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollectionToken_;
  edm::EDGetTokenT<reco::RecoToSimCollection> recoToSimAssociationToken_;
  edm::EDGetTokenT<TrackingVertexCollection> trackingVertexCollectionToken_;
  edm::EDGetTokenT<edm::View<reco::Vertex>> vtxToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> pathLengthToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> momentumToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> timeToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> sigmatimeToken_;

  edm::EDGetTokenT<edm::ValueMap<float>> t0PidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> t0SafePidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> sigmat0SafePidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> trackMVAQualToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> sigmatofpiToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> sigmatofkToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> sigmatofpToken_;

  edm::EDGetTokenT<edm::ValueMap<float>> tmtdToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> tofPiToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> tofKToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> tofPToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> probPiToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> probKToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> probPToken_;

};

//
// constructors and destructor
//
GNNInputs::GNNInputs(const edm::ParameterSet& iConfig)
    : theTTBToken(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      histName_(iConfig.getUntrackedParameter<std::string>("histName")),
      maxD0Sig_(iConfig.getParameter<double>("maxD0Significance")),
      maxD0Error_(iConfig.getParameter<double>("maxD0Error")),
      maxDzError_(iConfig.getParameter<double>("maxDzError")),
      minPt_(iConfig.getParameter<double>("minPt")),
      maxEta_(iConfig.getParameter<double>("maxEta")),
      maxNormChi2_(iConfig.getParameter<double>("maxNormalizedChi2")),
      minSiLayers_(iConfig.getParameter<int>("minSiliconLayersWithHits")),
      minPxLayers_(iConfig.getParameter<int>("minPixelLayersWithHits")),
      trackweightTh_(iConfig.getParameter<double>("trackweightTh")),	
      gnn_pt(-100.),
      gnn_eta(-100.),
      gnn_phi(-100.),
      gnn_z_pca(-100.),
      gnn_dz(-100.),
      gnn_t_pi(-100.),
      gnn_t_k(-100.),
      gnn_t_p(-100.),
      gnn_sigma_t0safe(-100.),
      gnn_t0safe(-100.),
      gnn_t0pid(-100.),
      gnn_mva_qual(-100.),
      gnn_btlMatchChi2(-100.),
      gnn_btlMatchTimeChi2(-100.),
      gnn_etlMatchChi2(-100.),
      gnn_etlMatchTimeChi2(-100.),
      gnn_pathLength(-100.),
      gnn_npixBarrel(-100),
      gnn_npixEndcap(-100),
      gnn_mtdTime(-100.),
      gnn_is_matched_tp(-100),
      gnn_sim_vertex_evID(-100),
      gnn_sim_vertex_BX(-100),
      gnn_sim_vertex_z(-100.),
      gnn_sim_vertex_t(-100.),
      gnn_tp_tEst(-100.),
      gnn_tp_pdgId(-100),
      gnn_probPi(-100.),
      gnn_probK(-100.),
      gnn_probP(-100.){
  RecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTracks"));
  RecBeamSpotToken_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("offlineBS"));
  trackingParticleCollectionToken_ = consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("SimTag"));
  recoToSimAssociationToken_ = consumes<reco::RecoToSimCollection>(iConfig.getParameter<edm::InputTag>("TPtoRecoTrackAssoc"));
  trackingVertexCollectionToken_ = consumes<TrackingVertexCollection>(iConfig.getParameter<edm::InputTag>("SimTag"));
  Rec4DVerToken_ = consumes<edm::View<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("offline3DPV"));
  vtxToken_ = consumes<edm::View<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("offline3DPV")) ;
  pathLengthToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("pathLengthSrc"));
  momentumToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("momentumSrc"));
  timeToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("timeSrc"));
  sigmatimeToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmaSrc"));
  t0PidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0PID"));
  t0SafePidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0SafePID"));
  sigmat0SafePidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmat0SafePID"));
  trackMVAQualToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("trackMVAQual"));
  tmtdToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tmtd"));
  tofPiToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tofPi"));
  tofKToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tofK"));
  tofPToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tofP"));
  probPiToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("probPi"));
  probKToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("probK"));
  probPToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("probP"));
  sigmatofpiToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmatofpiSrc"));
  sigmatofkToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmatofkSrc"));
  sigmatofpToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmatofpSrc"));
  btlMatchChi2Token_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("btlMatchChi2Src"));
  btlMatchTimeChi2Token_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("btlMatchTimeChi2Src"));
  etlMatchChi2Token_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("etlMatchChi2Src"));
  etlMatchTimeChi2Token_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("etlMatchTimeChi2Src"));
  npixBarrelToken_ = consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("npixBarrelSrc"));
  npixEndcapToken_ = consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("npixEndcapSrc"));
}

GNNInputs::~GNNInputs() {}

//
// member functions
//

const edm::Ref<std::vector<TrackingParticle>>* GNNInputs::getAnyMatchedTP(
    const reco::TrackBaseRef& recoTrack) {
  auto found = r2s_->find(recoTrack);
  // reco track not matched to any TP
  if (found == r2s_->end())
    return nullptr;

  //matched TP equal to any TP
  for (const auto& tp : found->val) {
    return &tp.first;
  }

  // reco track not matched to any TP from vertex
  return nullptr;
}

std::pair<const edm::Ref<std::vector<TrackingParticle>>*, int> GNNInputs::getMatchedTP(
    const reco::TrackBaseRef& recoTrack, const TrackingVertexRef& vsim) {
  auto found = r2s_->find(recoTrack);

  // reco track not matched to any TP (fake tracks)
  if (found == r2s_->end())
    return std::make_pair(nullptr, -1);

  // matched TP equal to any TP of a given sim vertex
  for (const auto& tp : found->val) {
    if (std::find_if(vsim->daughterTracks_begin(), vsim->daughterTracks_end(), [&](const TrackingParticleRef& vtp) {
          return tp.first == vtp;
        }) != vsim->daughterTracks_end())
      return std::make_pair(&tp.first, 0);
    // matched TP not associated to any daughter track of a given sim vertex but having the same eventID (track from secondary vtx)
    else if (tp.first->eventId().bunchCrossing() == vsim->eventId().bunchCrossing() &&
             tp.first->eventId().event() == vsim->eventId().event()) {
      return std::make_pair(&tp.first, 1);
    }
    // matched TP not associated to any sim vertex of a given simulated event (PU track)
    else {
      return std::make_pair(&tp.first, 2);
    }
  }

  // reco track not matched to any TP from vertex
  return std::make_pair(nullptr, -1);
}

std::vector<GNNInputs::simPrimaryVertex> GNNInputs::getSimPVs(
    const edm::Handle<TrackingVertexCollection>& tVC) {
  std::vector<GNNInputs::simPrimaryVertex> simpv;
  int current_event = -1;
  int s = -1;
  for (TrackingVertexCollection::const_iterator v = tVC->begin(); v != tVC->end(); ++v) {

    // We keep only the first vertex from all the events at BX=0.
    if (v->eventId().bunchCrossing() != 0)
      continue;
    if (v->eventId().event() != current_event) {
      current_event = v->eventId().event();
    } else {
      continue;
    }
    s++;
    if (std::abs(v->position().z()) > 1000)
      continue;  // skip junk vertices

    // could be a new vertex, check  all primaries found so far to avoid multiple entries
    int key = std::distance(tVC->begin(), v);
    simPrimaryVertex sv(v->position().x(), v->position().y(), v->position().z(), v->position().t());
    sv.eventId = v->eventId();
    sv.sim_vertex = TrackingVertexRef(tVC, key);
    sv.OriginalIndex = s;

     for (TrackingParticleRefVector::iterator iTrack = v->daughterTracks_begin(); iTrack != v->daughterTracks_end();
         ++iTrack) {
      assert((**iTrack).eventId().bunchCrossing() == 0);
    }
    simPrimaryVertex* vp = nullptr;  // will become non-NULL if a vertex is found and then point to it
    for (std::vector<simPrimaryVertex>::iterator v0 = simpv.begin(); v0 != simpv.end(); v0++) {
      if ((sv.eventId == v0->eventId) && (std::abs(sv.x - v0->x) < 1e-5) && (std::abs(sv.y - v0->y) < 1e-5) &&
          (std::abs(sv.z - v0->z) < 1e-5)) {
        vp = &(*v0);
        break;
      }
    }

    if (!vp) {
      // this is a new vertex, add it to the list of sim-vertices
      simpv.push_back(sv);
      vp = &simpv.back();
    }

  }  // End of for loop on tracking vertices

  // In case of no simulated vertices, break here
  if (simpv.empty())
    return simpv;
   // Calculate distance of vertices from LV of the same eventId and BX

  return simpv;
}



std::vector<GNNInputs::recoPrimaryVertex> GNNInputs::getRecoPVs(
    const edm::Handle<edm::View<reco::Vertex>>& tVC) {
  std::vector<GNNInputs::recoPrimaryVertex> recopv;
  int r = -1;
  for (auto v = tVC->begin(); v != tVC->end(); ++v) {
    r++;
    // Skip junk vertices
    if (std::abs(v->z()) > 1000)
      continue;
    if (v->isFake() || !v->isValid())
      continue;

    recoPrimaryVertex sv(v->position().x(), v->position().y(), v->position().z());
    sv.recVtx = &(*v);
    sv.recVtxRef = reco::VertexBaseRef(tVC, std::distance(tVC->begin(), v));

    sv.OriginalIndex = r;
    sv.ndof = v->ndof();
    // this is a new vertex, add it to the list of reco-vertices
    recopv.push_back(sv);
    GNNInputs::recoPrimaryVertex* vp = &recopv.back();
     for (auto iTrack = v->tracks_begin(); iTrack != v->tracks_end(); ++iTrack) {
      auto momentum = (*(*iTrack)).innerMomentum();
      if (momentum.mag2() == 0)
        momentum = (*(*iTrack)).momentum();
      vp->pt += std::sqrt(momentum.perp2());
      vp->ptsq += (momentum.perp2());
      vp->nRecoTrk++;

      auto matched = r2s_->find(*iTrack);
      if (matched != r2s_->end()) {
        vp->num_matched_sim_tracks++;
      }

    }  // End of for loop on daughters reconstructed tracks
  }    // End of for loop on tracking vertices

  // In case of no reco vertices, break here
  if (recopv.empty())
    return recopv;

  // Now compute the closest distance in z between all reconstructed vertex
  // first initialize
  auto prev_z = recopv.back().z;
  for (recoPrimaryVertex& vreco : recopv) {
    vreco.closest_vertex_distance_z = std::abs(vreco.z - prev_z);
    prev_z = vreco.z;
  }
for (std::vector<recoPrimaryVertex>::iterator vreco = recopv.begin(); vreco != recopv.end(); vreco++) {
    std::vector<recoPrimaryVertex>::iterator vreco2 = vreco;
    vreco2++;
    for (; vreco2 != recopv.end(); vreco2++) {
      double distance = std::abs(vreco->z - vreco2->z);
      // need both to be complete
      vreco->closest_vertex_distance_z = std::min(vreco->closest_vertex_distance_z, distance);
      vreco2->closest_vertex_distance_z = std::min(vreco2->closest_vertex_distance_z, distance);
    }
  }
  return recopv;
}

double GNNInputs::timeFromTrueMass(double mass, double pathlength, double momentum, double time) {
  if (time > 0 && pathlength > 0 && mass > 0) {
    double gammasq = 1. + momentum * momentum / (mass * mass);
    double v = c_ * std::sqrt(1. - 1. / gammasq);  // cm / ns
    double t_est = time - (pathlength / v);

    return t_est;
  } else {
    return -1;
  }
}

bool GNNInputs::passTrackFilter(const reco::TransientTrack& tk) {

  if (!tk.stateAtBeamLine().isValid())
    return false;
  bool IPSigCut = (tk.stateAtBeamLine().transverseImpactParameter().significance() < maxD0Sig_) &&
                  (tk.stateAtBeamLine().transverseImpactParameter().error() < maxD0Error_) &&
                  (tk.track().dzError() < maxDzError_);
  bool pTCut = tk.impactPointState().globalMomentum().transverse() > minPt_;
  bool etaCut = std::fabs(tk.impactPointState().globalMomentum().eta()) < maxEta_;
  bool normChi2Cut = tk.normalizedChi2() < maxNormChi2_;
  bool nPxLayCut = tk.hitPattern().pixelLayersWithMeasurement() >= minPxLayers_;
  bool nSiLayCut = tk.hitPattern().trackerLayersWithMeasurement() >= minSiLayers_;

  return IPSigCut && pTCut && etaCut && normChi2Cut && nPxLayCut && nSiLayCut; 

}


// ------------ method called for each event  ------------
void GNNInputs::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using edm::Handle;
  using edm::View;
  using std::cout;
  using std::endl;
  using std::vector;
  using namespace reco;
  TH1D* hist_common = new TH1D("histCommonTracks", "Common Tracks Count", 100, 0, 50);
  std::string treeName = "tree_" + std::to_string(iEvent.id().event());
  TTree* tree = fs_->make<TTree>(treeName.c_str(), "Tree for tracks");
  tree->Branch("gnn_pt", &gnn_pt, "gnn_pt/D");
  tree->Branch("gnn_eta", &gnn_eta, "gnn_eta/D");
  tree->Branch("gnn_phi", &gnn_phi, "gnn_phi/D");
  tree->Branch("gnn_z_pca", &gnn_z_pca, "gnn_z_pca/D");
  tree->Branch("gnn_dz", &gnn_dz, "gnn_dz/D");
  tree->Branch("gnn_t_pi", &gnn_t_pi, "gnn_t_pi/D");
  tree->Branch("gnn_t_k", &gnn_t_k, "gnn_t_k/D");
  tree->Branch("gnn_t_p", &gnn_t_p, "gnn_t_p/D");
  tree->Branch("gnn_sigma_t0safe", &gnn_sigma_t0safe, "gnn_sigma_t0safe/D");
  tree->Branch("gnn_sigma_tmtd", &gnn_sigma_tmtd, "gnn_sigma_tmtd/D");
  tree->Branch("gnn_t0safe", &gnn_t0safe, "gnn_t0safe/D");
  tree->Branch("gnn_t0pid", &gnn_t0pid, "gnn_t0pid/D");
  tree->Branch("gnn_mva_qual", &gnn_mva_qual, "gnn_mva_qual/D");
  tree->Branch("gnn_btlMatchChi2", &gnn_btlMatchChi2, "gnn_btlMatchChi2/D");
  tree->Branch("gnn_btlMatchTimeChi2", &gnn_btlMatchTimeChi2, "gnn_btlMatchTimeChi2/D");
  tree->Branch("gnn_etlMatchChi2", &gnn_etlMatchChi2, "gnn_etlMatchChi2/D");
  tree->Branch("gnn_etlMatchTimeChi2", &gnn_etlMatchTimeChi2, "gnn_etlMatchTimeChi2/D");
  tree->Branch("gnn_pathLength", &gnn_pathLength, "gnn_pathLength/D");
  tree->Branch("gnn_npixBarrel", &gnn_npixBarrel, "gnn_npixBarrel/I");
  tree->Branch("gnn_npixEndcap", &gnn_npixEndcap, "gnn_npixEndcap/I");
  tree->Branch("gnn_mtdTime", &gnn_mtdTime, "gnn_mtdTime/D");
  tree->Branch("gnn_is_matched_tp", &gnn_is_matched_tp, "gnn_is_matched_tp/I");
  tree->Branch("gnn_sim_vertex_evID", &gnn_sim_vertex_evID, "gnn_sim_vertex_evID/I");
  tree->Branch("gnn_sim_vertex_BX", &gnn_sim_vertex_BX, "gnn_sim_vertex_BX/I");
  tree->Branch("gnn_sim_vertex_z", &gnn_sim_vertex_z, "gnn_sim_vertex_z/D");
  tree->Branch("gnn_sim_vertex_t", &gnn_sim_vertex_t, "gnn_sim_vertex_t/D");
  tree->Branch("gnn_sim_vertex_index", &gnn_sim_vertex_index, "gnn_sim_vertex_index/I");
  tree->Branch("gnn_tp_tEst", &gnn_tp_tEst, "gnn_tp_tEst/D");
  tree->Branch("gnn_tp_pdgId", &gnn_tp_pdgId, "gnn_tp_pdgId/I");
  tree->Branch("gnn_probPi", &gnn_probPi, "gnn_probPi/D");
  tree->Branch("gnn_probK", &gnn_probK, "gnn_probK/D");
  tree->Branch("gnn_probP", &gnn_probP, "gnn_probP/D");
  tree->Branch("gnn_sigma_tof_Pi", &gnn_sigma_tof_Pi, "gnn_sigma_tof_Pi/D");
  tree->Branch("gnn_sigma_tof_K", &gnn_sigma_tof_K, "gnn_sigma_tof_K/D");
  tree->Branch("gnn_sigma_tof_P", &gnn_sigma_tof_P, "gnn_sigma_tof_P/D");
  tree->Branch("gnn_trk_chi2", &gnn_trk_chi2, "gnn_trk_chi2/D");
  tree->Branch("gnn_trk_ndof", &gnn_trk_ndof, "gnn_trk_ndof/D");
  tree->Branch("gnn_trk_validhits", &gnn_trk_validhits, "gnn_trk_validhits/I");
  
  /*TH2D *hist = new TH2D("hist", "2D Histogram; t; z", 500, -7.0, 7.0, 500, -10, 10);
  auto hist = std::make_unique<TH2D>(
        Form("hist_event_%llu", iEvent.id().event()),
        Form("2D Histogram for Event %llu; vtx Z; vtx T", iEvent.id().event()),
        10000, -20.0, 20, 1000, -20, 20);
*/
  edm::Handle<reco::TrackCollection> tracksH;
  iEvent.getByToken(RecTrackToken_, tracksH);

  const auto& theB = &iSetup.getData(theTTBToken);
  std::vector<reco::TransientTrack> t_tks;

  edm::Handle<TrackingParticleCollection> TPCollectionH;
  iEvent.getByToken(trackingParticleCollectionToken_, TPCollectionH);
  if (!TPCollectionH.isValid())
    edm::LogWarning("GNNInputs") << "TPCollectionH is not valid";

  edm::Handle<reco::RecoToSimCollection> recoToSimH;
  iEvent.getByToken(recoToSimAssociationToken_, recoToSimH);
  if (recoToSimH.isValid())
    r2s_ = recoToSimH.product();
  else
    edm::LogWarning("GNNInputs") << "recoToSimH is not valid";

  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> BeamSpotH;
  iEvent.getByToken(RecBeamSpotToken_, BeamSpotH);
  if (!BeamSpotH.isValid())
    edm::LogWarning("GNNInputs") << "BeamSpotH is not valid";
  beamSpot = *BeamSpotH;

  edm::Handle<TrackingVertexCollection> TVCollectionH;
  iEvent.getByToken(trackingVertexCollectionToken_, TVCollectionH);
  if (!TVCollectionH.isValid())
    edm::LogWarning("GNNInputs") << "TVCollectionH is not valid";

  const auto& trackingVertices = *TVCollectionH;

  edm::Handle<edm::View<reco::Vertex>> tVReco;
  iEvent.getByToken(vtxToken_, tVReco);
  std::vector<simPrimaryVertex> simpv;  // a list of simulated primary MC vertices
  simpv = getSimPVs(TVCollectionH);

  std::vector<recoPrimaryVertex> recopv;  // a list of reconstructed primary MC vertices
  edm::Handle<edm::View<reco::Vertex>> recVtxs;
  iEvent.getByToken(Rec4DVerToken_, recVtxs);

  if (!recVtxs.isValid())
    edm::LogWarning("GNNInputs") << "recVtxs is not valid";
  recopv = getRecoPVs(recVtxs);



  const auto& pathLength = iEvent.get(pathLengthToken_);
  const auto& momentum = iEvent.get(momentumToken_);
  const auto& time = iEvent.get(timeToken_);
  const auto& sigmatimemtd = iEvent.get(sigmatimeToken_);
  const auto& t0Pid = iEvent.get(t0PidToken_);
  const auto& t0Safe = iEvent.get(t0SafePidToken_);
  const auto& sigmat0Safe = iEvent.get(sigmat0SafePidToken_);
  const auto& mtdQualMVA = iEvent.get(trackMVAQualToken_);
  const auto& tMtd = iEvent.get(tmtdToken_);
  const auto& tofPi = iEvent.get(tofPiToken_);
  const auto& tofK = iEvent.get(tofKToken_);
  const auto& tofP = iEvent.get(tofPToken_);
  const auto& probPi = iEvent.get(probPiToken_);
  const auto& probK = iEvent.get(probKToken_);
  const auto& probP = iEvent.get(probPToken_);
  const auto& sigmatofpi = iEvent.get(sigmatofpiToken_);
  const auto& sigmatofk = iEvent.get(sigmatofkToken_);
  const auto& sigmatofp = iEvent.get(sigmatofpToken_);

  const auto& btlMatchChi2 = iEvent.get(btlMatchChi2Token_);
  const auto& btlMatchTimeChi2 = iEvent.get(btlMatchTimeChi2Token_);
  const auto& etlMatchChi2 = iEvent.get(etlMatchChi2Token_);
  const auto& etlMatchTimeChi2 = iEvent.get(etlMatchTimeChi2Token_);
  const auto& npixBarrel = iEvent.get(npixBarrelToken_);
  const auto& npixEndcap = iEvent.get(npixEndcapToken_);


  // build TransientTracks, needed for TrackFilter
  t_tks = (*theB).build(tracksH, beamSpot, t0Safe, sigmat0Safe);
  for (edm::View<reco::Vertex>::const_iterator iv = recVtxs->begin(); iv != recVtxs->end(); ++iv) {
  //for (unsigned int iv = 0; iv < recopv.size(); iv++) {
    const reco::Vertex* vertex = &(*iv); //recopv.at(iv).recVtx;
     unsigned int iv_index = std::distance(recVtxs->begin(), iv); 
     double vzsim = vertex->z();

     //for (unsigned int v2 = iv+1; v2 < recopv.size(); v2++) {
        for (edm::View<reco::Vertex>::const_iterator v2 = iv + 1; v2 != recVtxs->end(); ++v2) {
       	const reco::Vertex* vertex_v2 = &(*v2); //recopv.at(v2).recVtx;
	unsigned int v2_index = std::distance(recVtxs->begin(), v2);
	double vzsim_v2 = vertex_v2->z();
        int commonTrackCount = 0;
        std::cout << "Number of tracks for vertex " << iv_index << ": " 
                      << vertex->tracksSize() << std::endl;
            std::cout << "Number of tracks for vertex " << v2_index << ": " 
                      << vertex_v2->tracksSize() << std::endl;

      for (auto iTrack = vertex->tracks_begin(); iTrack != vertex->tracks_end(); ++iTrack) {
	 for (auto iTrack_v2 = vertex_v2->tracks_begin(); iTrack_v2 != vertex_v2->tracks_end(); ++iTrack_v2) {
        /*if (vertex->trackWeight(*iTrack) < trackweightTh_ && (vertex_v2->trackWeight(*iTrack_v2) < trackweightTh_))        {
          continue;
        }*/
       if (iv_index == 4 and v2_index == 88) {
        
	       std::cout << "RECHECK Number of tracks for vertex " << iv_index << ": "
                      << vertex->tracksSize() << " vzsim "<<vzsim<<std::endl;
               std::cout << "Number of tracks for vertex " << v2_index << ": "
                      << vertex_v2->tracksSize() <<" vzsim_v2 "<<vzsim_v2<<std::endl;
           std::cout<<" (*iTrack)->vz()  "<<(*iTrack)->vz()<<" (*iTrack_v2)->vz() "<<(*iTrack_v2)->vz()<<std::endl;

       }
          if ((*iTrack) == (*iTrack_v2)) {
		  std::cout<<" (*iTrack)->eta() "<<(*iTrack)->eta()<< " (*iTrack_v2)->eta() "<<(*iTrack_v2)->eta()<<std::endl;
                    commonTrackCount++;
		    std::cout<<"commonTrackCount "<<commonTrackCount<<std::endl;
         hist_common->Fill(commonTrackCount);
        }
       }  // RecoTracks loop
     }// RecoTracks2
  }
  }   
   int i = 0; 
  // Fill TTree with input variables for GNN
  for (std::vector<reco::TransientTrack>::const_iterator itk = t_tks.begin(); itk != t_tks.end(); itk++) {
    if (passTrackFilter(*itk)){
      reco::TrackBaseRef trackref = (*itk).trackBaseRef();
     
      gnn_pt = (*itk).track().pt();
      gnn_eta = (*itk).track().eta();
      gnn_phi = (*itk).track().phi();
      gnn_z_pca = (*itk).track().vz();
      gnn_dz = (*itk).track().dzError();
      gnn_t_pi = tMtd[trackref] - tofPi[trackref];
      gnn_t_k = tMtd[trackref] - tofK[trackref];
      gnn_t_p = tMtd[trackref] - tofP[trackref];
      gnn_sigma_t0safe = sigmat0Safe[trackref];
      gnn_sigma_tmtd = sigmatimemtd[trackref];
      gnn_t0safe = t0Safe[trackref];
      gnn_t0pid = t0Pid[trackref];
      gnn_mva_qual = mtdQualMVA[trackref];
      gnn_btlMatchChi2 = btlMatchChi2[trackref];
      gnn_btlMatchTimeChi2 = btlMatchTimeChi2[trackref];
      gnn_etlMatchChi2 = etlMatchChi2[trackref];
      gnn_etlMatchTimeChi2 = etlMatchTimeChi2[trackref];
      gnn_pathLength = pathLength[trackref];
      gnn_npixBarrel = npixBarrel[trackref];
      gnn_npixEndcap = npixEndcap[trackref];
      gnn_mtdTime = tMtd[trackref];
      gnn_probPi = probPi[trackref];
      gnn_probK = probK[trackref];
      gnn_probP = probP[trackref];
      gnn_sigma_tof_Pi  = sigmatofpi[trackref];
      gnn_sigma_tof_K  = sigmatofk[trackref];
      gnn_sigma_tof_P  = sigmatofp[trackref];
      gnn_trk_chi2 = (*itk).track().chi2();
      gnn_trk_ndof = (*itk).track().ndof();
      gnn_trk_validhits = (*itk).track().numberOfValidHits();


// ################################# 
      auto anytp_info = getAnyMatchedTP(trackref);
      if (anytp_info != nullptr) {
         gnn_is_matched_tp = 1;
         double anytp_mass = (*anytp_info)->mass();
         gnn_tp_tEst = timeFromTrueMass(anytp_mass, pathLength[trackref], momentum[trackref], time[trackref]);
         gnn_tp_pdgId = std::abs((*anytp_info)->pdgId());
         gnn_sim_vertex_z  = (*anytp_info)->parentVertex()->position().z();
         gnn_sim_vertex_t  = (*anytp_info)->parentVertex()->position().t() * simUnit_;
         gnn_sim_vertex_evID = (*anytp_info)->parentVertex()->eventId().event();
         gnn_sim_vertex_BX = (*anytp_info)->parentVertex()->eventId().bunchCrossing();


	 TrackingVertexRef parentVertexRef = (*anytp_info)->parentVertex();

         // Retrieve the index of the parent vertex
         int vertexIndex = parentVertexRef.key();
	 gnn_sim_vertex_index = parentVertexRef.key();
	 const TrackingVertex& vertex = trackingVertices[vertexIndex];
	 //std::cout << "Vertex Index: " << vertexIndex << std::endl;
	 //cout<<" Z from index "<<vertex.position().z()<<" Time from index "<<vertex.position().t() * simUnit_<<endl;
//	cout<<" i "<<i<<" gnn_sim_vertex_z "<<gnn_sim_vertex_z<<" gnn_sim_vertex_t "<<gnn_sim_vertex_t<<" gnn_sim_vertex_evID "<<gnn_sim_vertex_evID<<" gnn_sim_vertex_BX "<<gnn_sim_vertex_BX<<endl;
//	         hist->Fill(gnn_sim_vertex_z, gnn_sim_vertex_t);
      }else{
         gnn_is_matched_tp = 0;
      }
      i++; 
      tree->Fill();
    }
  } // loop on tracks

hist_common->Write();
///histograms.push_back(std::move(hist));
eventTrees_[iEvent.id().event()] = tree;  
}

// ------------ method called once each job just before starting event loop  ------------
void GNNInputs::beginJob() {
  // please remove this method if not needed
}

// ------------ method called once each job just after ending the event loop  ------------
void GNNInputs::endJob() {
  // please remove this method if not needed
/*  for (auto& hist : histograms) {
        hist->Write();
    }*/
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void GNNInputs::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("TPtoRecoTrackAssoc", edm::InputTag("trackingParticleRecoTrackAsssociation"));
  desc.add<edm::InputTag>("SimTag", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("offlineBS", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("trackAssocSrc", edm::InputTag("trackExtenderWithMTD:generalTrackassoc"))->setComment("Association between General and MTD Extended tracks");
  desc.add<edm::InputTag>("offline3DPV", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("pathLengthSrc", edm::InputTag("trackExtenderWithMTD:generalTrackPathLength"));
  desc.add<edm::InputTag>("momentumSrc", edm::InputTag("trackExtenderWithMTD:generalTrackp"));
  desc.add<edm::InputTag>("tmtd", edm::InputTag("trackExtenderWithMTD:generalTracktmtd"));
  desc.add<edm::InputTag>("timeSrc", edm::InputTag("trackExtenderWithMTD:generalTracktmtd"));
  desc.add<edm::InputTag>("sigmaSrc", edm::InputTag("trackExtenderWithMTD:generalTracksigmatmtd"));
  desc.add<edm::InputTag>("t0PID", edm::InputTag("tofPID:t0"));
  desc.add<edm::InputTag>("sigmat0PID", edm::InputTag("tofPID:sigmat0"));
  desc.add<edm::InputTag>("t0SafePID", edm::InputTag("tofPID:t0safe"));
  desc.add<edm::InputTag>("sigmat0SafePID", edm::InputTag("tofPID:sigmat0safe"));
  desc.add<edm::InputTag>("trackMVAQual", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"));
  desc.add<edm::InputTag>("tofPi", edm::InputTag("trackExtenderWithMTD:generalTrackTofPi"));
  desc.add<edm::InputTag>("tofK", edm::InputTag("trackExtenderWithMTD:generalTrackTofK"));
  desc.add<edm::InputTag>("tofP", edm::InputTag("trackExtenderWithMTD:generalTrackTofP"));
  desc.add<edm::InputTag>("probPi", edm::InputTag("tofPID:probPi"));
  desc.add<edm::InputTag>("probK", edm::InputTag("tofPID:probK"));
  desc.add<edm::InputTag>("probP", edm::InputTag("tofPID:probP"));
  desc.add<edm::InputTag>("sigmatofpiSrc", edm::InputTag("trackExtenderWithMTD:generalTrackSigmaTofPi"));
  desc.add<edm::InputTag>("sigmatofkSrc", edm::InputTag("trackExtenderWithMTD:generalTrackSigmaTofK"));
  desc.add<edm::InputTag>("sigmatofpSrc", edm::InputTag("trackExtenderWithMTD:generalTrackSigmaTofP"));
  desc.add<edm::InputTag>("btlMatchChi2Src", edm::InputTag("trackExtenderWithMTD", "btlMatchChi2"));
  desc.add<edm::InputTag>("btlMatchTimeChi2Src", edm::InputTag("trackExtenderWithMTD", "btlMatchTimeChi2"));
  desc.add<edm::InputTag>("etlMatchChi2Src", edm::InputTag("trackExtenderWithMTD", "etlMatchChi2"));
  desc.add<edm::InputTag>("etlMatchTimeChi2Src", edm::InputTag("trackExtenderWithMTD", "etlMatchTimeChi2"));
  desc.add<edm::InputTag>("npixBarrelSrc", edm::InputTag("trackExtenderWithMTD", "npixBarrel"));
  desc.add<edm::InputTag>("npixEndcapSrc", edm::InputTag("trackExtenderWithMTD", "npixEndcap"));
  desc.addUntracked<std::string>("histName","file.root");
  desc.add<double>("maxD0Significance", 4.0);
  desc.add<double>("maxD0Error", 1.0);
  desc.add<double>("maxDzError", 1.0);
  desc.add<double>("minPt", 0.0);
  desc.add<double>("maxEta", 2.4);
  desc.add<double>("maxNormalizedChi2", 10.0);
  desc.add<int>("minSiliconLayersWithHits", 5);
  desc.add<int>("minPixelLayersWithHits", 2);
  desc.add<double>("trackweightTh", 0.5);
  descriptions.addDefault(desc);

}

//define this as a plug-in
DEFINE_FWK_MODULE(GNNInputs);
