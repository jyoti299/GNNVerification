// -*- C++ -*-
//
// Package:    SimStudy/SimEdgeweights
// Class:      SimEdgeweights
//
/**\class SimEdgeweights SimEdgeweights.cc SimStudy/SimEdgeweights/plugins/SimEdgeweights.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jyoti Babbar
//         Created:  Thu, 21 Nov 2024 10:07:25 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "/gfsvol01/cms/users/jbabbar/work/container/master/Sim_weights/CMSSW_14_1_0_pre7/src/SimStudy/SimEdgeweights/interface/TracksGraph.h"

// reco track and vertex
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

// TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/PrimaryVertexProducer/interface/HITrackFilterForPVFinding.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"
//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

using reco::TrackCollection;
using namespace cms::Ort;
const int FEATURE_SHAPE_GNN_V1 = 2;
const int FEATURE_SHAPE_MLP = 2;
const int NUM_EDGE_FEATURES = 2;
enum class DNNVersion {
    GNN_V1,
    MLP_EDGE_FEATURES,
    MLP_NO_EDGE_FEATURES
};
std::unordered_map<std::string, DNNVersion> DNN_VERSION_MAP = {
    {"gnn_v1", DNNVersion::GNN_V1},
    {"mlp_edge_features", DNNVersion::MLP_EDGE_FEATURES},
    {"mlp_no_edge_features", DNNVersion::MLP_NO_EDGE_FEATURES}
};
std::unordered_map<DNNVersion, std::pair<std::vector<std::string>, int>> dnnInputConfigurations = {
    {DNNVersion::GNN_V1, {{"features", "edge_index", "edge_features"}, FEATURE_SHAPE_GNN_V1}},
    {DNNVersion::MLP_EDGE_FEATURES, {{"features", "edge_index", "edge_features"}, FEATURE_SHAPE_MLP}},
    {DNNVersion::MLP_NO_EDGE_FEATURES, {{"features", "edge_index"}, FEATURE_SHAPE_MLP}}
};
//class SimEdgeweights : public edm::one::EDAnalyzer<edm::one::SharedResources> {
class SimEdgeweights : public edm::stream::EDAnalyzer<edm::GlobalCache<ONNXRuntime>> {
typedef math::XYZTLorentzVector LorentzVector;
    struct simPrimaryVertex {
    simPrimaryVertex(double x1, double y1, double z1, double t1, int k1)
        : x(x1),
          y(y1),
          z(z1),
          t(t1),
          key(k1),
          LV_distance_z(-1.){};
    double x, y, z, t;
    int key;
    TrackingVertexRef sim_vertex;
    int OriginalIndex = -1;
    EncodedEventId eventId;
    bool is_LV;
    double LV_distance_z;

  };
public:
  SimEdgeweights(const edm::ParameterSet& iConfig,  const ONNXRuntime* onnxRuntime);// cms::Ort::ONNXRuntime const* onnxRuntime);
  ~SimEdgeweights() override;
  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);//  cms::Ort::ONNXRuntime const* onnxRuntime);
  static void globalEndJob(const ONNXRuntime*);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  std::vector<SimEdgeweights::simPrimaryVertex> getSimPVs(const edm::Handle<TrackingVertexCollection>&);
  const edm::Ref<std::vector<TrackingParticle>>* getAnyMatchedTP(const reco::TrackBaseRef&);
  edm::Service<TFileService> fs_;
  std::unique_ptr<TrackGraph>  produce_tracks_graph(const std::vector<reco::TransientTrack>& transientTrack) const;
   
  // ----------member data ---------------------------
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> theTTBToken;
  TrackFilterForPVFindingBase* theTrackFilter;

  static constexpr unsigned int NOT_MATCHED = 66666;
  static constexpr double simUnit_ = 1e9;     //sim time in s while reco time in ns
  static constexpr double c_ = 2.99792458e1;  //c in cm/ns
  edm::EDGetTokenT<TrackCollection> tracksToken_;  //used to select what tracks to read from configuration file
  const ONNXRuntime* onnxRuntime_;
  edm::EDGetTokenT<edm::ValueMap<float> > trkTimesToken;
  edm::EDGetTokenT<edm::ValueMap<float> > trkTimeResosToken;
  const reco::RecoToSimCollection* r2s_;
  const reco::SimToRecoCollection* s2r_;
  edm::EDGetTokenT<reco::TrackCollection> RecTrackToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> RecVertexToken_;
  edm::EDGetTokenT<reco::BeamSpot> RecBeamSpotToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollectionToken_;
  edm::EDGetTokenT<TrackingVertexCollection> trackingVertexCollectionToken_;
  edm::EDGetTokenT<reco::SimToRecoCollection> simToRecoAssociationToken_;
  edm::EDGetTokenT<reco::RecoToSimCollection> recoToSimAssociationToken_;
  edm::EDGetTokenT<edm::View<reco::Vertex>> Rec4DVerToken_;
  edm::EDGetTokenT<edm::ValueMap<int>> trackAssocToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SimEdgeweights::SimEdgeweights(const edm::ParameterSet& iConfig, const ONNXRuntime* onnxRuntime = nullptr)
    : theTTBToken(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTracks"))),
       onnxRuntime_(onnxRuntime) {
  trkTimesToken = consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("TrackTimesLabel"));
    trkTimeResosToken = consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("TrackTimeResosLabel"));  
  trackingParticleCollectionToken_ =
      consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("SimTag"));
  trackingVertexCollectionToken_ = consumes<TrackingVertexCollection>(iConfig.getParameter<edm::InputTag>("SimTag"));
  simToRecoAssociationToken_ =
      consumes<reco::SimToRecoCollection>(iConfig.getParameter<edm::InputTag>("TPtoRecoTrackAssoc"));
  recoToSimAssociationToken_ =
      consumes<reco::RecoToSimCollection>(iConfig.getParameter<edm::InputTag>("TPtoRecoTrackAssoc"));
  RecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTracks"));
  RecVertexToken_ = consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("inputTagV"));
  RecBeamSpotToken_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("offlineBS"));
  Rec4DVerToken_ = consumes<edm::View<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("offline4DPV"));
  trackAssocToken_ = consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("trackAssocSrc"));
  std::string trackSelectionAlgorithm =
      iConfig.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<std::string>("algorithm");
  if (trackSelectionAlgorithm == "filter") {
    theTrackFilter = new TrackFilterForPVFinding(iConfig.getParameter<edm::ParameterSet>("TkFilterParameters"));
  } else if (trackSelectionAlgorithm == "filterWithThreshold") {
    theTrackFilter = new HITrackFilterForPVFinding(iConfig.getParameter<edm::ParameterSet>("TkFilterParameters"));
  } else {
    edm::LogWarning("MVATrainingNtuple: unknown track selection algorithm: " + trackSelectionAlgorithm);
  }
  //now do what ever initialization is needed
}

std::unique_ptr<ONNXRuntime> SimEdgeweights::initializeGlobalCache(const edm::ParameterSet &iConfig) {// const ONNXRuntime* onnxRuntime){
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("onnxModelPath").fullPath());
}
void SimEdgeweights::globalEndJob(const ONNXRuntime* onnxRuntime) {}
SimEdgeweights::~SimEdgeweights() {
	 if (theTrackFilter)
    delete theTrackFilter;
}

//
// member functions
//
const edm::Ref<std::vector<TrackingParticle>>* SimEdgeweights::getAnyMatchedTP(
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

std::vector<SimEdgeweights::simPrimaryVertex> SimEdgeweights::getSimPVs(
    const edm::Handle<TrackingVertexCollection>& tVC) {
  std::vector<SimEdgeweights::simPrimaryVertex> simpv;
  std::set<std::pair<int, int>> processedEvents;  // Set to store (event ID, bunch crossing) pairs
  int current_event = -1;
  int s = -1;
  for (TrackingVertexCollection::const_iterator v = tVC->begin(); v != tVC->end(); ++v) {

     if (v->eventId().bunchCrossing() != 0)
      continue;
    bool is_LV = true;
    std::pair<unsigned int, int> eventBunchPair = std::make_pair(v->eventId().event(), v->eventId().bunchCrossing());
     if (std::abs(v->position().z()) > 1000)
      continue;  // skip junk vertices
     if (v->eventId().event() != current_event) {
      current_event = v->eventId().event();
    } else {
      continue;
    }
    // Skip the vertex if this event ID and bunch crossing pair has already been processed
    if (processedEvents.find(eventBunchPair) != processedEvents.end()) {
      is_LV = false;
    }

    // Mark this event ID and bunch crossing pair as processed
    processedEvents.insert(eventBunchPair);
    s++;

    // could be a new vertex, check  all primaries found so far to avoid multiple entries
    int key = std::distance(tVC->begin(), v);
    simPrimaryVertex sv(v->position().x(), v->position().y(), v->position().z(), v->position().t(), key);
    sv.sim_vertex = TrackingVertexRef(tVC, key);
    sv.OriginalIndex = s;
    sv.is_LV = is_LV;
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
  for (unsigned int iev = 0; iev < simpv.size(); iev++) {
    simpv.at(iev).LV_distance_z = std::abs(simpv.at(0).z - simpv.at(iev).z);
  }

  return simpv;
}
std::unique_ptr<TrackGraph>  SimEdgeweights::produce_tracks_graph(const std::vector<reco::TransientTrack>& transientTracks) const {
    std::vector<Node> allNodes;

for (size_t i = 0; i < transientTracks.size(); ++i) {
            const reco::TransientTrack& ttrack_node = transientTracks[i];
            float zPosition = ttrack_node.track().vz();
        allNodes.emplace_back(i, zPosition);  // Create a new Node with index i
    }
    for (size_t i = 0; i < transientTracks.size(); ++i) {
        const reco::TransientTrack& ttrack_node = transientTracks[i];

        for (size_t j = i + 1; j < transientTracks.size(); ++j) {
            const reco::TransientTrack& ttrack = transientTracks[j];

            double neighbour_cut = std::abs(ttrack_node.track().vz() - ttrack.track().vz());
            if (neighbour_cut < 0.3) {
                // Add inner connections between i and j
                   allNodes[i].addInner(j);  // Track i adds track j as an inner node
                   allNodes[j].addInner(i);   // Track j adds track i as an inner node
                  // std::cout << "Adding neighbor to track i: " <<i<<" neighbour "<<j<<" with Z position: "
                    //  << allNodes[j].getZPosition() << std::endl;
                }
               }
              }

     auto resultGraph = std::make_unique<TrackGraph>(allNodes);
    return resultGraph;
}

// ------------ method called for each event  ------------
void SimEdgeweights::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  using edm::Handle;
  using std::vector;
  using namespace reco;
     cms::Ort::FloatArrays data;
    std::vector<float> features;
    std::vector<float> edge_features;
    std::vector<int> selected_track_indices;  // To store indices of selected tracks
std::vector<reco::TransientTrack> selected_tracks;  // To store the selected tracks
std::vector<float> selected_features;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<float> edges_src;
    std::vector<float> edges_dst;
    std::vector<double> gnn_sim_vertex_z;
  edm::Handle<reco::TrackCollection> tracksH;
  iEvent.getByToken(RecTrackToken_, tracksH);

  const auto& theB = &iSetup.getData(theTTBToken);
  std::vector<reco::TransientTrack> t_tks;

  edm::Handle<TrackingParticleCollection> TPCollectionH;
  iEvent.getByToken(trackingParticleCollectionToken_, TPCollectionH);
  if (!TPCollectionH.isValid())
    edm::LogWarning("SimEdgeweights") << "TPCollectionH is not valid";

  edm::Handle<reco::RecoToSimCollection> recoToSimH;
  iEvent.getByToken(recoToSimAssociationToken_, recoToSimH);
  if (recoToSimH.isValid())
    r2s_ = recoToSimH.product();
  else
    edm::LogWarning("SimEdgeweights") << "recoToSimH is not valid";

  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> BeamSpotH;
  iEvent.getByToken(RecBeamSpotToken_, BeamSpotH);
  if (!BeamSpotH.isValid())
    edm::LogWarning("SimEdgeweights") << "BeamSpotH is not valid";
  beamSpot = *BeamSpotH;

   edm::Handle<TrackingVertexCollection> TVCollectionH;
  iEvent.getByToken(trackingVertexCollectionToken_, TVCollectionH);
  if (!TVCollectionH.isValid())
    edm::LogWarning("SimEdgeweights") << "TVCollectionH is not valid";

  std::vector<simPrimaryVertex> simpv;
  simpv = getSimPVs(TVCollectionH);

  const auto& trackAssoc = iEvent.get(trackAssocToken_);

  std::vector<reco::Vertex> vertices;
  edm::Handle<std::vector<reco::Vertex>> RecVertexHandle;
  iEvent.getByToken(RecVertexToken_, RecVertexHandle);
  vertices = *RecVertexHandle;
  auto const& trackTimeResos_ = iEvent.get(trkTimeResosToken);
    auto trackTimes_ = iEvent.get(trkTimesToken);
  
  DNNVersion versionEnum = DNN_VERSION_MAP["gnn_v1"];
    // Access the input configuration based on the DNN version
    std::vector<std::string> input_names;
    int shapeFeatures = 0;
    auto dnnConfig = dnnInputConfigurations.find(versionEnum);
    if (dnnConfig != dnnInputConfigurations.end()) {
        // Valid DNN version found, retrieve input names and shape features
        input_names = dnnConfig->second.first;
        shapeFeatures = dnnConfig->second.second;
    } else {
        // Invalid DNN version
        edm::LogError("Vertex Producer") << "Architecture not defined: " ;
    }
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  // if the SetupData is always needed
  // if need the ESHandle to check if the SetupData was there or not
#endif

   for (unsigned int iev = 0; iev < simpv.size(); iev++) {
        auto vsim = simpv.at(iev).sim_vertex;
	bool selectedLV = simpv.at(iev).eventId.bunchCrossing() == 0 && simpv.at(iev).eventId.event() == 0;
        double vzsim = simpv.at(iev).z;
        double vtsim = simpv.at(iev).t * simUnit_;
        
       for (TrackingVertex::tp_iterator iTP = vsim->daughterTracks_begin(); 
         iTP != vsim->daughterTracks_end(); ++iTP) { 
	       const TrackingParticle& track = **iTP;
	       //auto momentum = (*(*iTP)).momentum();
	       float trackvz = (**iTP).vz();
	       //std::cout << "  Track vz: " << trackvz  << std::endl;
   }
  }
     t_tks = (*theB).build(tracksH, beamSpot, trackTimes_, trackTimeResos_);
      int N = t_tks.size();
     int k =0;

     //auto trackgraph = produce_tracks_graph(t_tks);
     //TrackGraph *trkgrp = trackgraph.get();
     for (size_t i = 0; i < t_tks.size() ; i++) {
      const reco::TransientTrack& ttrack1 = t_tks[i]; 

      reco::TrackBaseRef trackref = ttrack1.trackBaseRef();
       bool foundMatch = false;
       auto anytp_info = getAnyMatchedTP(trackref);
        if (anytp_info != nullptr) {
          TrackingVertexRef parentVertexRef = (*anytp_info)->parentVertex();
	   for (const auto& vsim : simpv) {
               if (vsim.sim_vertex == parentVertexRef) {
		foundMatch = true;       
                gnn_sim_vertex_z.push_back(vsim.z);
                double z_pca = ttrack1.track().vz();
                double dz = ttrack1.track().dzError();
		selected_tracks.push_back(ttrack1);  // Store the selected track
                selected_track_indices.push_back(i);
                features.push_back(z_pca);
                features.push_back(dz);
                k++; 
       }
    }
  }
	if (!foundMatch) {
                   features.push_back(-999);
                    features.push_back(-999);
         }
 }
   auto trackgraph = produce_tracks_graph(selected_tracks);
   TrackGraph *trkgrp = trackgraph.get();
    for (size_t i = 0; i < selected_tracks.size(); ++i) {
    const reco::TransientTrack& ttrack1 = selected_tracks[i];
     for (size_t j = i+1; j < selected_tracks.size(); ++j) {
        if (i == j) continue;  // Skip self-pairing (no self-edges)

        const reco::TransientTrack& ttrack2 = selected_tracks[j];
        	    
//	for (size_t j = 0; j < t_tks.size() ; ++j){
          //const reco::TransientTrack& ttrack2 = t_tks[j];
                  reco::TrackBaseRef trackref1 = ttrack1.trackBaseRef();
	  reco::TrackBaseRef trackref2 = ttrack2.trackBaseRef();
	   auto anytp_info1 = getAnyMatchedTP(trackref1);
	  auto anytp_info2 = getAnyMatchedTP(trackref2);
	  if (anytp_info1 != nullptr && anytp_info2 != nullptr) {
          TrackingVertexRef parentVertexRef1 = (*anytp_info1)->parentVertex();		  
          TrackingVertexRef parentVertexRef2 = (*anytp_info2)->parentVertex();


    if ((parentVertexRef1 == parentVertexRef2)) {
        std::cout << "Parent Vertex 1 z-position: " << parentVertexRef1->position() << std::endl;
        std::cout << "Parent Vertex 2 z-position: " << parentVertexRef2->position() << std::endl;
        edges_src.push_back(static_cast<float>(selected_track_indices[j]));
        edges_dst.push_back(static_cast<float>(selected_track_indices[i]));
	double zpca_diff = std::abs(ttrack1.track().vz() - ttrack2.track().vz());
        double dz_sign = std::abs(ttrack1.track().vz() - ttrack2.track().vz())/sqrt(((ttrack1.track().dzError())*(ttrack1.track().dzError()))+((ttrack2.track().dzError())*(ttrack2.track().dzError())));
	std::cout<<" True edges indices "<<selected_track_indices[i]<<" and j "<<selected_track_indices[j]<<" z[i] "<<ttrack1.track().vz()<<" z[j] "<<ttrack2.track().vz()<<" zpca_diff "<<zpca_diff<<" dz_sign "<<dz_sign<<std::endl;
	 edge_features.push_back(zpca_diff);
         edge_features.push_back(dz_sign); 
	    }
          } }
	  }
	  
	auto numEdges = static_cast<int>(edges_src.size());
	data.clear();
	std::cout<<" N "<<N<<" features "<<features.size()<<"edge_features"<<edge_features.size()<<" shapeFeatures "<<shapeFeatures<<" k "<<k<<" edges_src "<<edges_src.size()<<" edges_sdst "<<edges_dst.size()<<std::endl;
    input_shapes.push_back({1, N, shapeFeatures});
    data.emplace_back(features);
    input_shapes.push_back({1, 2, numEdges});
    data.emplace_back(edges_src);
    for (auto &dst : edges_dst)
    {
            data.back().push_back(dst);
    }
    input_shapes.push_back({1, numEdges, NUM_EDGE_FEATURES});
    data.emplace_back(edge_features);

// run prediction
    auto edge_predictions = onnxRuntime_->run(input_names, data,input_shapes)[0];
    std::cout<<" edge_predictions size "<<edge_predictions.size()<<std::endl; 
       for (int i = 0; i < static_cast<int>(edge_predictions.size()); i++){
             //std::cout<<" i "<<i<<" edge_predictions[i] "<<edge_predictions[i]<<std::endl;
             trkgrp->setEdgeWeight(data[1][i], data[1][numEdges + i], edge_predictions[i]);
    }
   auto connected_components = trkgrp->findSubComponents(0.90);
   std::cout<<" connected_components "<<connected_components.size()<<std::endl;
} //analyze  


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SimEdgeweights::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("TrackTimeResosLabel", edm::InputTag("dummy_default"));                         // 4D only
  desc.add<edm::InputTag>("TrackTimesLabel", edm::InputTag("dummy_default"));
  desc.add<edm::InputTag>("TPtoRecoTrackAssoc", edm::InputTag("trackingParticleRecoTrackAsssociation"));
  desc.add<edm::InputTag>("mtdTracks", edm::InputTag("trackExtenderWithMTD"));
  desc.add<edm::InputTag>("inputTracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("inputTagV", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("SimTag", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("offlineBS", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("offline4DPV", edm::InputTag("offlinePrimaryVertices4D"));
  desc.add<edm::InputTag>("trackAssocSrc", edm::InputTag("trackExtenderWithMTD:generalTrackassoc"))
      ->setComment("Association between General and MTD Extended tracks");
  { 
    edm::ParameterSetDescription psd0;
    HITrackFilterForPVFinding::fillPSetDescription(psd0);  // extension of TrackFilterForPVFinding
    desc.add<edm::ParameterSetDescription>("TkFilterParameters", psd0);
  }
  desc.add<edm::FileInPath>("onnxModelPath", edm::FileInPath("/gfsvol01/cms/users/jbabbar/work/container/master/Sim_weights/CMSSW_14_1_0_pre7/src/SimStudy/SimEdgeweights/data/model_v6version_PULV_TTbarPU_4Oct_bidirection_th0p3.onnx"))->setComment("Path to GNN (as ONNX model)");
  descriptions.addDefault(desc);

}

//define this as a plug-in
DEFINE_FWK_MODULE(SimEdgeweights);
