/*
 * Evaluates the BDT score of ldmx-sw 1.7.0/1.7.1 data
 * and a few other output variables. As a single
 * ROOT macro. 
 *
 * This also includes a reimplementation of the reconstruction
 * of all ECal BDT Variables! (I couldn't run re-recon easily)
 *
 * Adapted from EventProc/src/EcalVetoProcessor.cxx in ldmx-sw 1.7.0
 * and
 * EcalVeto/bdtTreeMaker.py in IncandelaLab/LDMX-scripts
 *
 * Contact: Erik Wallin
 */

/*
 * Since this macro is really long I include an index,
 * so that you can search and jump to the right place,
 * skipping uninteresting sections. 
 *
 * Index:
 * Section Utility Functions
 * Section Analysis Start
 * Section Output Variable Definitions
 * Section Event Loop Start
 * Section Recoil Trajectories
 * Section Track Veto
 * Section HCal Veto
 * Section Containment Radii Definitions
 * Section ECal Hit Loop With Found Electron Trajectory
 * Section ECal Hit Loop Without Found Electron Trajectory
 * Section Additional Noise Generation
 * Section Containment Variable Calculations
 * Section Other BDT Variable Calculations
 * Section Evaluate BDT
 */

R__LOAD_LIBRARY(/opt/ldmx-v1.7.0/lib/libEvent.so)

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TPython.h"
#include "TObject.h"
#include "TMath.h"
#include "TRandom.h"

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <math.h>

//Path to the header files must be specified before this macro starts.
//E.g. with gROOT->ProcessLine(".include PATH");
//in rootlogon.C in the same directory

#include "Event/EcalHit.h"
#include "Event/CalorimeterHit.h"
#include "Event/SimParticle.h"
#include "Event/SimTrackerHit.h"
#include "Event/EcalVetoResult.h"
#include "Event/FindableTrackResult.h"
#include "Event/HcalVetoResult.h"
#include "Event/HcalHit.h"
#include "Event/TrackerVetoResult.h"
#include "Event/EventHeader.h"

// Section Utility Functions

//Converts a vector of BDT variables to a string, so that it can be sent to Python and be evaluated in the xgboost BDT. 
//Note: There is no loss of float precision.
TString vectorToPredCMD(std::vector<double> bdtFeatures) {
    TString featuresStrVector = "[[";
    for (int i = 0; i < bdtFeatures.size(); i++) {
        std::stringstream s;
        s << std::fixed << std::setprecision(10) << bdtFeatures[i];
        featuresStrVector += s.str();
        if (i < bdtFeatures.size() - 1)
        featuresStrVector += ",";
    }
    featuresStrVector += "]]";
    TString cmd = "float(model.predict(xgb.DMatrix(np.array(" + featuresStrVector + ")))[0])";

    return cmd;
}

//Translate ECal cell ID to an index of the 'mcid' vector
int findCellIndex(std::vector<int> ids, int cellid){
    for(int i=0; i < 2779; i++){
        if(ids.at(i) == cellid){
            return i;
        }
    }
    return -1;
}


//Returns a random number of a normal distribution beyond x >= 4
//Samples uniform numbers in a sub-set of the unit square
//Then normal-distributing them with the Box-Muller Transform
Double_t sample4SigmaTail(TRandom3* rng){
    while(true){
        Double_t u1 = rng->Uniform(0,TMath::Exp(-pow(4,2)/2));
        //Here the limit 0.05 is chosen to minimize the area that includes numbers below 4 sigma
        //If it is set to 1 (or 0.5), this method produces true normal distributed numbers
        //With a performance cost
        Double_t u2 = rng->Uniform(0,0.05);
        Double_t z = TMath::Sqrt(-2*TMath::Log(u1))*TMath::Cos(2*3.14159*u2);
        if(z > 4){
            return z;
        }
    }
}

// Section Analysis Start

/*
 * Evaluates BDT score, track veto and HCal veto of data. 
 * if noise = true, extra Gaussian noise is added to ECal Cells with a std. deviation of noiseFactor times the original std. deviation.
 * completeOutput = true writes all BDT variables to the output ROOT-file. E.g. for when you want to train the BDT.
 * reconstruct = True, discards all BDT variables saved in the reconstructed input files and calculates them from scratch. Also useful if your data is not reconstructed. 
 */

void eval(TString infilename, TString outfilename, bool noise = false, double noiseFactor = 1.25, bool completeOutput = false, bool reconstruct = false){
    //bool reconstruct = false;

    std::cout << "Starting BDT evaluation" << std::endl;

    //Python side imports, as xgboost will be evaluated
    //interop:ed with Python
    TPython::Exec("print(\'Loading xgboost BDT\')"); 
    TPython::Exec("import xgboost as xgb");
    TPython::Exec("import numpy as np");
    TPython::Exec("import pickle as pkl");
    TPython::Exec("print(\"xgboost version: \" + xgb.__version__)");
    TPython::Exec("model = pkl.load(open(\'august19_0/august19_0_weights.pkl\','rb'))");

    TPython::Exec("print(\"Testing loaded model with zeros input\")");
    TPython::Exec("pred = float(model.predict(xgb.DMatrix(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))[0])");

    TPython::Exec("print(\"Test score: \" + str(pred))");

    //Load cell position map
    TPython::Exec("cellMap = np.loadtxt(\'cellmodule.txt\')");
    std::vector<int> mcid;
    std::vector<float> cellX;
    std::vector<float> cellY;

    //Lazily run the loop inside Python 
    //to not bother with string concatenation in C++
    TPython::Exec("i_ = 0");
    for(int i=0; i < 2779; i++){
        mcid.push_back(TPython::Eval("cellMap[i_,0]"));
        cellX.push_back(TPython::Eval("cellMap[i_,1]"));
        cellY.push_back(TPython::Eval("cellMap[i_,2]"));

        TPython::Exec("i_ += 1");
    }
   
    //Build nearest neighbours map for isolated hit searches
    std::map<int,std::vector<int>> NNMap;

    double nCellsWide = 23;
    double moduler = 85;
    double moduleR = moduler*(2./sqrt(3.));
    double cellr = moduleR/(nCellsWide-1./3.);

    for(int j = 0; j < 2779; j++){
        for(int k = 0; k < 2779; k++){
            double dist = sqrt(pow(cellX.at(k)-cellX.at(j),2)+pow(cellY.at(k)-cellY.at(j),2));
            if(dist > cellr && dist <= 3.*cellr){
                NNMap[j].push_back(k);
            }
        }
    }
 
    //Layer positions
    std::vector<double> layerZs{223.8000030517578, 226.6999969482422, 233.0500030517578, 237.4499969482422, 245.3000030517578, 251.1999969482422, 260.29998779296875,266.70001220703125, 275.79998779296875, 282.20001220703125, 291.29998779296875, 297.70001220703125, 306.79998779296875, 313.20001220703125,322.29998779296875, 328.70001220703125, 337.79998779296875, 344.20001220703125, 353.29998779296875, 359.70001220703125, 368.79998779296875,375.20001220703125, 384.29998779296875, 390.70001220703125, 403.29998779296875, 413.20001220703125, 425.79998779296875, 435.70001220703125,448.29998779296875, 458.20001220703125, 470.79998779296875, 480.70001220703125, 493.29998779296875, 503.20001220703125};

    //Containment radii for 8 GeV
    //std::vector<double> radius_recoil_68_p_0_1000_theta_0_6{9.65155163169527,9.029949540117707,8.169116380219359,7.26878332423302,5.723387467629167,5.190678018534044,5.927290663506518,6.182560329200212,7.907549398117859,8.606100542857211,10.93381822596916,12.043201938160239,14.784548371508041,16.102403056546482,18.986402399412817,20.224453740305716,23.048820910305643,24.11202594672678,26.765135236851666,27.78700483852502,30.291794353801293,31.409870873194464,33.91006482486666,35.173073672355926,38.172422630271,40.880288341493205,44.696485719120005,49.23802839743545,53.789910813378675,60.87843355562641,66.32931132415688,75.78117972604727,86.04697356716805,96.90360704034346};
    //std::vector<double> radius_recoil_68_p_1000_end_theta_0_6{9.65155163169527,9.029949540117707,8.169116380219359,7.26878332423302,5.723387467629167,5.190678018534044,5.927290663506518,6.182560329200212,7.907549398117859,8.606100542857211,10.93381822596916,12.043201938160239,14.784548371508041,16.102403056546482,18.986402399412817,20.224453740305716,23.048820910305643,24.11202594672678,26.765135236851666,27.78700483852502,30.291794353801293,31.409870873194464,33.91006482486666,35.173073672355926,38.172422630271,40.880288341493205,44.696485719120005,49.23802839743545,53.789910813378675,60.87843355562641,66.32931132415688,75.78117972604727,86.04697356716805,96.90360704034346};
    //std::vector<double> radius_recoil_68_theta_6_15{6.486368894455214,6.235126063894043,6.614742647173138,7.054111110170857,7.6208431229479645,8.262931570498493,10.095697703256274,11.12664183734125,13.463274649564859,14.693527904936063,17.185557959405358,18.533873226278285,21.171912124279075,22.487821335146958,25.27214729142235,26.692900194943586,29.48033347163334,30.931911179461117,33.69749728369263,35.35355537189422,37.92163028706617,40.08541101327325,42.50547781670488,44.42600915526537,48.18838292957783,50.600428280254235,55.85472906972822,60.88022977643599,68.53506382625108,73.0547148939902,78.01129860152466,90.91421661272666,104.54696678290463,116.90671501444335};
    //std::vector<double> radius_recoil_68_theta_15_end{7.218181823299591,7.242577749118457,9.816977116964644,12.724324104744532,17.108322705113288,20.584866353828193,25.036863838363544,27.753201816619153,32.08174405069556,34.86092888550297,39.56748303616661,43.37808998888681,48.50525488266305,52.66203291220487,58.00763047516536,63.028585648616584,69.21745026096245,74.71857224945907,82.15269906028466,89.1198060894434,95.15548897621329,103.91086738998598,106.92403611582472,115.76216727231979,125.72534759956525,128.95688953061537,140.84273174274335,151.13069543119798,163.87399183389545,171.8032189173357,186.89216628021853,200.19270470457505,219.32987417488016,236.3947885046377};

    //The same containment radii but in one long vector

    std::vector<double> radii{9.65155163169527,9.029949540117707,8.169116380219359,7.26878332423302,5.723387467629167,5.190678018534044,5.927290663506518,6.182560329200212,7.907549398117859,8.606100542857211,10.93381822596916,12.043201938160239,14.784548371508041,16.102403056546482,18.986402399412817,20.224453740305716,23.048820910305643,24.11202594672678,26.765135236851666,27.78700483852502,30.291794353801293,31.409870873194464,33.91006482486666,35.173073672355926,38.172422630271,40.880288341493205,44.696485719120005,49.23802839743545,53.789910813378675,60.87843355562641,66.32931132415688,75.78117972604727,86.04697356716805,96.90360704034346,9.65155163169527,9.029949540117707,8.169116380219359,7.26878332423302,5.723387467629167,5.190678018534044,5.927290663506518,6.182560329200212,7.907549398117859,8.606100542857211,10.93381822596916,12.043201938160239,14.784548371508041,16.102403056546482,18.986402399412817,20.224453740305716,23.048820910305643,24.11202594672678,26.765135236851666,27.78700483852502,30.291794353801293,31.409870873194464,33.91006482486666,35.173073672355926,38.172422630271,40.880288341493205,44.696485719120005,49.23802839743545,53.789910813378675,60.87843355562641,66.32931132415688,75.78117972604727,86.04697356716805,96.90360704034346,6.486368894455214,6.235126063894043,6.614742647173138,7.054111110170857,7.6208431229479645,8.262931570498493,10.095697703256274,11.12664183734125,13.463274649564859,14.693527904936063,17.185557959405358,18.533873226278285,21.171912124279075,22.487821335146958,25.27214729142235,26.692900194943586,29.48033347163334,30.931911179461117,33.69749728369263,35.35355537189422,37.92163028706617,40.08541101327325,42.50547781670488,44.42600915526537,48.18838292957783,50.600428280254235,55.85472906972822,60.88022977643599,68.53506382625108,73.0547148939902,78.01129860152466,90.91421661272666,104.54696678290463,116.90671501444335,7.218181823299591,7.242577749118457,9.816977116964644,12.724324104744532,17.108322705113288,20.584866353828193,25.036863838363544,27.753201816619153,32.08174405069556,34.86092888550297,39.56748303616661,43.37808998888681,48.50525488266305,52.66203291220487,58.00763047516536,63.028585648616584,69.21745026096245,74.71857224945907,82.15269906028466,89.1198060894434,95.15548897621329,103.91086738998598,106.92403611582472,115.76216727231979,125.72534759956525,128.95688953061537,140.84273174274335,151.13069543119798,163.87399183389545,171.8032189173357,186.89216628021853,200.19270470457505,219.32987417488016,236.3947885046377};
    
    //Input file
    TFile* infile = new TFile(infilename);
    TTree* intree = (TTree*)infile->Get("LDMX_Events");
    
    Int_t nHits;
    Float_t summedDet;
    Float_t summedTightIso;
    Float_t maxCellDep;
    Float_t showerRMS;
    Float_t xStd;
    Float_t yStd;
    Float_t avgLayerHit;
    Float_t stdLayerHit;
    Int_t deepestLayerHit;

    //Input variables

    TClonesArray* ecalHits = new TClonesArray("ldmx::EcalHit");
    intree->SetBranchAddress("ecalDigis_recon",&ecalHits);

    TClonesArray* hcalHits = new TClonesArray("ldmx::HcalHit");
    intree->SetBranchAddress("hcalDigis_recon",&hcalHits);

    TClonesArray* simParticles = new TClonesArray("ldmx::SimParticle");
    intree->SetBranchAddress("SimParticles_sim", &simParticles);

    TClonesArray* simTrackerHits = new TClonesArray("ldmx::SimTrackerHit");
    intree->SetBranchAddress("EcalScoringPlaneHits_sim", &simTrackerHits);

    TClonesArray* targetSPHits = new TClonesArray("ldmx::SimTrackerHit");
    intree->SetBranchAddress("TargetScoringPlaneHits_sim",&targetSPHits);
    
    if(!reconstruct){
    TClonesArray* ecalVetoResult = new TClonesArray("ldmx::EcalVetoResult");
    intree->SetBranchAddress("EcalVeto_recon",&ecalVetoResult);
    }

    TClonesArray* findableTracks = new TClonesArray("ldmx::FindableTrackResult");
    intree->SetBranchAddress("FindableTracks_recon", &findableTracks);

    TClonesArray* hcalVetoResult = new TClonesArray("ldmx::HcalVetoResult");
    intree->SetBranchAddress("HcalVeto_recon",&hcalVetoResult);

    //TClonesArray* eventHeader = new TClonesArray("ldmx::EventHeader");
    //ldmx::EventHeader* eventHeader;
    //double* eventWeight;
    //intree->SetBranchAddress("EventHeader/weight_",&eventWeight);


    //TClonesArray* trackerVetoResult = new TClonesArray("ldmx::TrackerVetoResult");
    //intree->SetBranchAddress("TrackerVeto_recon",&trackerVetoResult);
    //TClonesArray* trackerVetoResultRerecon = new TClonesArray("ldmx::TrackerVetoResult");
    //intree->SetBranchAddress("TrackerVeto_rerecon",&trackerVetoResultRerecon);
    //bool isRerecon = false;

    //Why can't I manage to read this in pure ROOT...
    //While it is easily read on the Python side for some reason
    if(!reconstruct){
        TPython::Exec("ecalVetoRes = ROOT.TClonesArray(\"ldmx::EcalVetoResult\")");
    }
    TPython::Exec("evHeader = ROOT.ldmx.EventHeader()");
    TPython::Bind(intree,"intree");
    if(!reconstruct){
        TPython::Exec("intree.SetBranchAddress(\"EcalVeto_recon\",ecalVetoRes)");
    }
    TPython::Exec("intree.SetBranchAddress(\"EventHeader\",evHeader)");

    double eventWeight;

    //Output

    TFile* outfile = new TFile(outfilename, "RECREATE");
    TTree* outtree = new TTree("BDTree", "BDT Scores");
    
    // Section Output Variable Definitions

    double score;
    auto branch = outtree->Branch("Score", &score);

    bool passesTrackVeto;
    auto vetoBranch = outtree->Branch("passesTrackVeto", &passesTrackVeto);

    bool passesHcalVeto;
    auto hcalBranch = outtree->Branch("passesHcalVeto", &passesHcalVeto); 

    int maxPE;
    auto maxPEBranch = outtree->Branch("maxPE", &maxPE);

    int numOfFindableTracks;
    auto numOfFindableTracksBranch = outtree->Branch("numOfFindableTracks", &numOfFindableTracks);

    double recoilP;
    auto recoilPBranch = outtree->Branch("recoilP", &recoilP);

    double recoilElectronP;
    auto recoilElectronPBranch = outtree->Branch("recoilElectronP", &recoilElectronP);

    double recoilPt;
    auto recoilPtBranch = outtree->Branch("recoilPt", &recoilPt);

    double frontEcal;
    auto frontEcalBranch = outtree->Branch("frontEcal", &frontEcal);

    int anomolousParticle;
    auto anomolousParticleBranch = outtree->Branch("anomolousParticle", &anomolousParticle);

    bool foundElectron;
    auto foundElectronBranch = outtree->Branch("foundElectron", &foundElectron);

    auto eventWeightBranch = outtree->Branch("eventWeight", &eventWeight);
   
    double ele68ContEnergy;
    double ele68x2ContEnergy;
    double ele68x3ContEnergy;
    double ele68x4ContEnergy;
    double ele68x5ContEnergy;
    double photon68ContEnergy;
    double photon68x2ContEnergy;
    double photon68x3ContEnergy;
    double photon68x4ContEnergy;
    double photon68x5ContEnergy;
    double overlap68ContEnergy;
    double overlap68x2ContEnergy;
    double overlap68x3ContEnergy;
    double overlap68x4ContEnergy;
    double overlap68x5ContEnergy;
    double outside68ContEnergy;
    double outside68x2ContEnergy;
    double outside68x3ContEnergy;
    double outside68x4ContEnergy;
    double outside68x5ContEnergy;
    double outside68ContNHits;
    double outside68x2ContNHits;
    double outside68x3ContNHits;
    double outside68x4ContNHits;
    double outside68x5ContNHits;
    double outside68ContXstd;       
    double outside68ContYstd;       
    double outside68x2ContXstd;       
    double outside68x2ContYstd;       
    double outside68x3ContXstd;       
    double outside68x3ContYstd;       
    double outside68x4ContXstd;       
    double outside68x4ContYstd;       
    double outside68x5ContXstd;       
    double outside68x5ContYstd;

    double ecalBackEnergy;
 
    auto summedDetBranch = outtree->Branch("summedDet", &summedDet);

    // Do not include all BDT variables in the output file if you do not want them
    if(completeOutput){
        auto ecalBackEnergyBranch = outtree->Branch("ecalBackEnergy", &ecalBackEnergy);
        auto nHitsBranch = outtree->Branch("nHits", &nHits);
        //auto summedDetBranch = outtree->Branch("summedDet", &summedDet);
        auto summedTightIsoBRanch = outtree->Branch("summedTightIso", &summedTightIso);
        auto maxCellDepBranch = outtree->Branch("maxCellDep", &maxCellDep);
        auto showerRMSBranch = outtree->Branch("showerRMS", &showerRMS);
        auto xStdBranch = outtree->Branch("xStd", &xStd);
        auto yStdBranch = outtree->Branch("yStd", &yStd);
        auto avgLayerHitBranch = outtree->Branch("avgLayerHit", &avgLayerHit);
        auto stdLayerHitBranch = outtree->Branch("stdLayerHit", &stdLayerHit);
        auto deepestLayerHitBranch = outtree->Branch("deepestLayerHit", &deepestLayerHit);
        auto ele68ContEnergyBranch = outtree->Branch("ele68ContEnergy",&ele68ContEnergy);
        auto ele68x2ContEnergyBranch = outtree->Branch("ele68x2ContEnergy",&ele68x2ContEnergy);
        auto ele68x3ContEnergyBranch = outtree->Branch("ele68x3ContEnergy",&ele68x3ContEnergy);
        auto ele68x4ContEnergyBranch = outtree->Branch("ele68x4ContEnergy",&ele68x4ContEnergy);
        auto ele68x5ContEnergyBranch = outtree->Branch("ele68x5ContEnergy",&ele68x5ContEnergy);
        auto photon68ContEnergyBranch = outtree->Branch("photon68ContEnergy",&photon68ContEnergy);
        auto photon68x2ContEnergyBranch = outtree->Branch("photon68x2ContEnergy",&photon68x2ContEnergy);
        auto photon68x3ContEnergyBranch = outtree->Branch("photon68x3ContEnergy",&photon68x3ContEnergy);
        auto photon68x4ContEnergyBranch = outtree->Branch("photon68x4ContEnergy",&photon68x4ContEnergy);
        auto photon68x5ContEnergyBranch = outtree->Branch("photon68x5ContEnergy",&photon68x5ContEnergy);
        auto outside68ContEnergyBranch = outtree->Branch("outside68ContEnergy",&outside68ContEnergy);
        auto outside68x2ContEnergyBranch = outtree->Branch("outside68x2ContEnergy",&outside68x2ContEnergy);
        auto outside68x3ContEnergyBranch = outtree->Branch("outside68x3ContEnergy",&outside68x3ContEnergy);
        auto outside68x4ContEnergyBranch = outtree->Branch("outside68x4ContEnergy",&outside68x4ContEnergy);
        auto outside68x5ContEnergyBranch = outtree->Branch("outside68x5ContEnergy",&outside68x5ContEnergy);
        auto outside68ContNHitsBranch = outtree->Branch("outside68ContNHits",&outside68ContNHits);
        auto outside68x2ContNHitsBranch = outtree->Branch("outside68x2ContNHits",&outside68x2ContNHits);
        auto outside68x3ContNHitsBranch = outtree->Branch("outside68x3ContNHits",&outside68x3ContNHits);
        auto outside68x4ContNHitsBranch = outtree->Branch("outside68x4ContNHits",&outside68x4ContNHits);
        auto outside68x5ContNHitsBranch = outtree->Branch("outside68x5ContNHits",&outside68x5ContNHits);
        auto outside68ContXstdBranch = outtree->Branch("outside68ContXstd",&outside68ContXstd);
        auto outside68ContYstdBranch = outtree->Branch("outside68ContYstd",&outside68ContYstd);
        auto outside68x2ContXstdBranch = outtree->Branch("outside68x2ContXstd",&outside68x2ContXstd);
        auto outside68x2ContYstdBranch = outtree->Branch("outside68x2ContYstd",&outside68x2ContYstd);
        auto outside68x3ContXstdBranch = outtree->Branch("outside68x3ContXstd",&outside68x3ContXstd);
        auto outside68x3ContYstdBranch = outtree->Branch("outside68x3ContYstd",&outside68x3ContYstd);
        auto outside68x4ContXstdBranch = outtree->Branch("outside68x4ContXstd",&outside68x4ContXstd);
        auto outside68x4ContYstdBranch = outtree->Branch("outside68x4ContYstd",&outside68x4ContYstd);
        auto outside68x5ContXstdBranch = outtree->Branch("outside68x5ContXstd",&outside68x5ContXstd);
        auto outside68x5ContYstdBranch = outtree->Branch("outside68x5ContYstd",&outside68x5ContYstd);
    }

    //Process events
    // Section Event Loop Start

    std::cout << "Processing " << intree->GetEntriesFast() << " events" << std::endl;

    //Seed set to 0 for random seed
    TRandom3* rng = new TRandom3(0);

    double noiseProb = noiseFactor * ROOT::Math::normal_cdf_c(4*noiseFactor*1506.32*0.13/33000, noiseFactor*1506.32*0.13/33000);
    std::cout << "noiseProb " << noiseProb << std::endl;

    int passedTrack = 0;
    int passedHcal = 0;
    int passed0_999 = 0;

    int passedAll = 0;

    //Hack warning: Because we are reading some things through Python interop
    //The event loop index is stored separately in the Python side
    //So we do not have to pass it as an argument everytime a Python cmd
    //needs it
    TPython::Exec("i=-1");
    for(int i=0; i<intree->GetEntriesFast(); i++){
    //for(int i=0; i<1; i++){

        TPython::Exec("intree.GetEntry(i); i +=1");
        //std::cout << i << std::endl;
     
        eventWeight = TPython::Eval("evHeader.getWeight()");
   
        nHits = 0; 
        summedDet = 0; 
        summedTightIso = 0; 
        maxCellDep = 0; 
        showerRMS = 0; 
        xStd = 0; 
        yStd = 0; 
        avgLayerHit = 0; 
        stdLayerHit = 0;
        deepestLayerHit = 0; 
        frontEcal = 0;

        Float_t xMean = 0;
        Float_t yMean = 0;
        Float_t wavgLayerHit = 0;

        intree->GetEntry(i);

        // Section Recoil Trajectories
        //Find electron and photon trajectories
        std::vector<double> pvec;
        std::vector<float> pos;
        foundElectron = false;
        bool foundTargetElectron = false;
        //Electron at target
        std::vector<double> pvec0;
        std::vector<float> pos0;

        std::vector<double> vetoP;

        double pz;

        int electronTrackID;

        for(int k=0; k < simParticles->GetEntriesFast(); k++){
            ldmx::SimParticle* particle = (ldmx::SimParticle*)simParticles->At(k);
            if(particle->getPdgID() == 11 && particle->getParentCount() == 0){

                electronTrackID = particle->getTrackID();

                for(int j=0; j < simTrackerHits->GetEntriesFast(); j++){
                    
                    ldmx::SimTrackerHit* hit = (ldmx::SimTrackerHit*)simTrackerHits->At(j);

                    //Doesn't work? Why?
                    //auto e = hit->getSimParticle();
                    //Match the TrackID instead
                    int hitTrackID = hit->getTrackID();

                    //if(hitTrackID == electronTrackID && hit->getLayerID() == 2){
                    //    vetoP = hit->getMomentum();
                    //}

                    //Why does the original script check if p.z > 0 ?
                    if(hitTrackID == electronTrackID && hit->getLayerID() == 1){
                        pvec = hit->getMomentum();
                        pos = hit->getPosition();
                        foundElectron = true;
                    }
                }
 
                for(int j=0; j < targetSPHits->GetEntriesFast(); j++){
                    ldmx::SimTrackerHit* hit = (ldmx::SimTrackerHit*)targetSPHits->At(j);

                    int hitTrackID = hit->getTrackID();

                    if(hitTrackID == electronTrackID && hit->getLayerID() == 2){
                        pvec0 = hit->getMomentum();
                        pos0 = hit->getPosition();
                        foundTargetElectron = true;
                    }
                }
            }
        }

        anomolousParticle = -1;
        for(int k=0; k < simParticles->GetEntriesFast(); k++){
            ldmx::SimParticle* particle = (ldmx::SimParticle*)simParticles->At(k);
            int particleTrackID = particle->getTrackID();
            for(int j=0; j < targetSPHits->GetEntriesFast(); j++){
                ldmx::SimTrackerHit* hit = (ldmx::SimTrackerHit*)targetSPHits->At(j);

                int hitTrackID = hit->getTrackID();

                if(hitTrackID == particleTrackID && hit->getLayerID() == 2){
                    //pvec0 = hit->getMomentum();
                    //pos0 = hit->getPosition();
                    //foundTargetElectron = true;
                    int id = particle->getPdgID();
                    if(id != 22 && id != 11  && id != -11){
                        anomolousParticle = id;
                    }
                }
            }
        }
    
        recoilPt = -1;
        recoilElectronP = -1;
        if(foundTargetElectron){
            recoilPt = TMath::Sqrt(pow(pvec0.at(0),2)+pow(pvec0.at(1),2));
            recoilElectronP = TMath::Sqrt(pow(pvec0.at(0),2)+pow(pvec0.at(1),2)+pow(pvec0.at(2),2));
        }       

        // Section Track Veto
        //Check track veto        
        numOfFindableTracks = 0;
        bool trackFindable = false; 
        std::vector<double> trackP;
        
        for(int k=0; k < findableTracks->GetEntriesFast(); k++){
            
            ldmx::FindableTrackResult* track = (ldmx::FindableTrackResult*)findableTracks->At(k);

            if(track->is4sFindable() || track->is3s1aFindable() || track->is2s2aFindable()){
                numOfFindableTracks += 1;

                ldmx::SimParticle* particle = track->getSimParticle();

                //Find momentum from SP hit
                for(int j=0; j < targetSPHits->GetEntriesFast(); j++){
                    ldmx::SimTrackerHit* hit = (ldmx::SimTrackerHit*)targetSPHits->At(j);

                    if((ldmx::SimParticle*)hit->getSimParticle() == particle && hit->getLayerID() == 2){
                        trackP = hit->getMomentum();
                        if(trackP.at(2) > 0){
                            trackFindable = true;
                        }
                    }
                }
            }
        }

        passesTrackVeto = false;

        recoilP = -1;
        
        if(numOfFindableTracks == 1 && trackFindable){
            recoilP = TMath::Sqrt(pow(trackP.at(0),2)+pow(trackP.at(1),2)+pow(trackP.at(2),2));
            if(recoilP < 2400){
                passesTrackVeto = true;
            }
        }

        if(passesTrackVeto){
            passedTrack += 1;
        }

        // Section HCal Veto
        //Hcal veto
        //Do not use the results from HcalVetoResult, it may not be correct

        maxPE = -1;
        for(int k = 0; k < hcalHits->GetEntriesFast(); k++){
            ldmx::HcalHit* h = (ldmx::HcalHit*)hcalHits->At(k);

            int pe = h->getPE();    

            if(h->getTime() >= 50) continue;
            if(h->getZ() > 4000) continue;

            //Check for photo-electrons on the back side of the hcal cell
            if(h->getSection() == 0 && h->getMinPE() < 1) continue; 

            if(pe > maxPE){
                maxPE = pe;
            }
        }

        passesHcalVeto = false;
        if(maxPE < 8){
            passedHcal += 1;
            passesHcalVeto = true;
        }        

        //Find trajectory of electron and (inferred) recoil photon 
        std::vector<double> elePosX;
        std::vector<double> elePosY;
        
        if(foundElectron){
            for(auto ite = layerZs.begin(); ite < layerZs.end(); ite++){
                elePosX.push_back(pos.at(0) + pvec.at(0)/pvec.at(2)*(*ite - pos.at(2)));
                elePosY.push_back(pos.at(1) + pvec.at(1)/pvec.at(2)*(*ite - pos.at(2)));
            }
        }
       
        std::vector<double> photonP;

        std::vector<double> photonPosX;
        std::vector<double> photonPosY;
       
        if(foundTargetElectron){
            photonP.push_back(-pvec0.at(0));
            photonP.push_back(-pvec0.at(1));
            photonP.push_back(8000-pvec0.at(2));
            for(auto ite = layerZs.begin(); ite < layerZs.end(); ite++){
                photonPosX.push_back(pos0.at(0) + photonP.at(0)/photonP.at(2)*(*ite - pos0.at(2)));
                photonPosY.push_back(pos0.at(1) + photonP.at(1)/photonP.at(2)*(*ite - pos0.at(2)));
            }
        }
       
        double theta = -1;
        double magnitude = -1;
 

        if(foundElectron){ 
            theta = TMath::ACos(pvec.at(2)/TMath::Sqrt(pow(pvec.at(0),2)+pow(pvec.at(1),2)+pow(pvec.at(2),2)));

            magnitude = TMath::Sqrt(pow(pvec.at(0),2)+pow(pvec.at(1),2)+pow(pvec.at(2),2));
        }

        // Section Containment Radii Definitions
 
        //Determine bin of electron phase space
        int ir = -1;
        if(theta < 6 && magnitude < 1000){
            ir = 1;
        }
        else if(theta < 6){
            ir = 2;
        }
        else if(theta < 15){
            ir = 3;
        }
        else{
            ir = 4;
        }
        //Photons always use the low-theta low-energy radii
        int ip = 1;

        //Calculate energy containment variables 
        
        ele68ContEnergy = 0;
        ele68x2ContEnergy = 0;
        ele68x3ContEnergy = 0;
        ele68x4ContEnergy = 0;
        ele68x5ContEnergy = 0;

        photon68ContEnergy = 0;
        photon68x2ContEnergy = 0;
        photon68x3ContEnergy = 0;
        photon68x4ContEnergy = 0;
        photon68x5ContEnergy = 0;

        overlap68ContEnergy = 0;
        overlap68x2ContEnergy = 0;
        overlap68x3ContEnergy = 0;
        overlap68x4ContEnergy = 0;
        overlap68x5ContEnergy = 0;

        outside68ContEnergy = 0;
        outside68x2ContEnergy = 0;
        outside68x3ContEnergy = 0;
        outside68x4ContEnergy = 0;
        outside68x5ContEnergy = 0;

        outside68ContNHits = 0;
        outside68x2ContNHits = 0;
        outside68x3ContNHits = 0;
        outside68x4ContNHits = 0;
        outside68x5ContNHits = 0;

        double outside68ContXmean = 0;
        double outside68x2ContXmean = 0;
        double outside68x3ContXmean = 0;
        double outside68x4ContXmean = 0;
        double outside68x5ContXmean = 0;

        double outside68ContYmean = 0;
        double outside68x2ContYmean = 0;
        double outside68x3ContYmean = 0;
        double outside68x4ContYmean = 0;
        double outside68x5ContYmean = 0;

        std::vector<double> ele68Totals{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        std::vector<double> photon68Totals{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        std::vector<double> overlap68Totals{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        std::vector<double> outside68Totals{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        std::vector<double> outside68NHits{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        std::vector<double> outside68Xmean{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        std::vector<double> outside68Ymean{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        double outside68WgtCentroidCoordsX = 0;
        double outside68WgtCentroidCoordsY = 0;
        double outside68x2WgtCentroidCoordsX = 0;
        double outside68x2WgtCentroidCoordsY = 0;
        double outside68x3WgtCentroidCoordsX = 0;
        double outside68x3WgtCentroidCoordsY = 0;
        double outside68x4WgtCentroidCoordsX = 0;
        double outside68x4WgtCentroidCoordsY = 0;
        double outside68x5WgtCentroidCoordsX = 0;
        double outside68x5WgtCentroidCoordsY = 0;


        //std::vector<std::vector<double>> outside68HitPositions;
        std::vector<double> outside68HitPositionsX;
        std::vector<double> outside68HitPositionsY;
        std::vector<int> outside68HitPositionsLayer;
        std::vector<double> outside68HitPositionsE;
        
        std::vector<double> outside68x2HitPositionsX;
        std::vector<double> outside68x2HitPositionsY;
        std::vector<int> outside68x2HitPositionsLayer;
        std::vector<double> outside68x2HitPositionsE;
        
        //std::vector<std::vector<double>> outside68x2HitPositions;
        
        std::vector<double> outside68x3HitPositionsX;
        std::vector<double> outside68x3HitPositionsY;
        std::vector<int> outside68x3HitPositionsLayer;
        std::vector<double> outside68x3HitPositionsE;

        //std::vector<std::vector<double>> outside68x3HitPositions;

        std::vector<double> outside68x4HitPositionsX;
        std::vector<double> outside68x4HitPositionsY;
        std::vector<int> outside68x4HitPositionsLayer;
        std::vector<double> outside68x4HitPositionsE;
        
        //std::vector<std::vector<double>> outside68x4HitPositions;
        
        std::vector<double> outside68x5HitPositionsX;
        std::vector<double> outside68x5HitPositionsY;
        std::vector<int> outside68x5HitPositionsLayer;
        std::vector<double> outside68x5HitPositionsE;
        
        //std::vector<std::vector<double>> outside68x5HitPositions;

        ecalBackEnergy = 0;

        
        //bool* occupiedCells = new bool[2779*34];
        bool occupiedCells[2779*34];
        std::fill(occupiedCells, occupiedCells+2779*34, false);
        bool occupiedCellsProjected[2779];
        std::fill(occupiedCellsProjected, occupiedCellsProjected+2779, false);

        std::vector<double> allX;
        std::vector<double> allY;
        std::vector<double> allE; 
        std::vector<double> allLayer;
        std::vector<int> allmcid;

        double wgtCentroidCoordsX = 0;
        double wgtCentroidCoordsY = 0;
        //double sumEdep = 0;

        // Section ECal Hit Loop With Found Electron Trajectory

        //Dont calculate photon containments even if that trajectory is found, when the electron ecal hit is missing
        if(foundElectron){
            for(int k=0; k < ecalHits->GetEntriesFast(); k++){
                ldmx::EcalHit* hit = (ldmx::EcalHit*)ecalHits->At(k);
                
                int rawID = hit->getID();

                //The raw ID is packed with:
                //4 bits for the subdetector, 8 cell layer bits, 3 module bits and 17 cell bits
                //On the form: ccccccccccccccccc mmm llllllll ssss
                //int layermask = 0xFF << 20;
                //int layer = (layermask & rawID) >> 20;
                int layermask = 0xFF << 4;
                int layer = (layermask & rawID) >> 4;            

                //int modulemask = 0x7 << 17;
                //int module = (modulemask & rawID) >> 17;
                int modulemask = 0x7 << 12;
                int module = (modulemask & rawID) >> 12;

                //int cellmask = 0x1FFFF;
                //int cell = rawID & 0x1FFFF;
                int cellmask = 0x1FFFF << 15;
                int cell = (rawID & cellmask) >> 15;

                int mcid_val = 10*cell+module;
                //int mcid_index_slow = findCellIndex(mcid, mcid_val);
                int mcid_index = (mcid_val % 10)*396 + (mcid_val - (mcid_val % 10))/10+ (mcid_val % 10);// + 1;
               // std::cout << mcid_val << ":" <<mcid_index_slow << ":" << mcid_index << std::endl;

                double hitX = cellX.at(mcid_index);
                double hitY = cellY.at(mcid_index);
                //std::cout << hitX <<":" << hitY << std::endl;

                double distanceEle = TMath::Sqrt(pow(hitX -elePosX.at(layer),2)+pow(hitY-elePosY.at(layer),2));

                double distancePhoton = -1;
                if(foundTargetElectron){
                    distancePhoton = TMath::Sqrt(pow(hitX -photonPosX.at(layer),2)+pow(hitY-photonPosY.at(layer),2));
                }

                

                double hitE = hit->getEnergy();
                
                allE.push_back(hitE);
                allX.push_back(hitX);
                allY.push_back(hitY);
                allLayer.push_back(layer);
                allmcid.push_back(mcid_index);
                occupiedCells[mcid_index+layer*2779] = true;
                occupiedCellsProjected[mcid_index] = true;

                //Add noise and check whether it falls below the threshold

                //Ignore all pure noise hits from the simulation
                //Added Gaussian noise to actual hits can not be removed
                if(noise){
                    if(hit->isNoise()){
                        continue;
                    }
                }

                if(noise){
                    hitE += rng->Gaus(0,noiseFactor*1506.32*0.13/33000);
     
                    if(!(hitE > 4*noiseFactor*1506.32*0.13/33000)){
                        continue;
                    }
                }
                else{
                    if(!(hitE > 0)){
                        continue;
                    }
                }

                //allE.push_back(hitE);
                //allX.push_back(hitX);
                //allY.push_back(hitY);
                //allLayer.push_back(layer);
                //allmcid.push_back(mcid_index);

                summedDet += hitE;
                nHits++;
                if(layer < 20){
                    frontEcal += hitE;
                }

                xMean += hitX*hitE;
                yMean += hitY*hitE;

                avgLayerHit += layer;
                wavgLayerHit += layer*hitE;

                wgtCentroidCoordsX += hitX*hitE;
                wgtCentroidCoordsY += hitY*hitE;

                //occupiedCells[mcid_index+layer*2779] = true;
                //occupiedCellsProjected[mcid_index] = true;
 
                if(layer >= 20){
                    ecalBackEnergy += hitE;
                }

                double ir_radius = radii.at((ir-1)*34+layer);

                if(distanceEle < ir_radius){
                    ele68ContEnergy += hitE;
                    ele68Totals.at(layer) += hitE;
                }
                else if(distanceEle < 2*ir_radius){
                    ele68x2ContEnergy += hitE;
                }
                else if(distanceEle < 3*ir_radius){
                    ele68x3ContEnergy += hitE;
                }
                else if(distanceEle < 4*ir_radius){
                    ele68x4ContEnergy += hitE;
                }
                else if(distanceEle < 5*ir_radius){
                    ele68x5ContEnergy += hitE;
                }
        
                double ip_radius = radii.at(layer);
                
                if(distancePhoton < ip_radius && distancePhoton > 0){
                    photon68ContEnergy += hitE;
                    photon68Totals.at(layer) += hitE;
                }
                else if(distancePhoton < 2*ip_radius && distancePhoton > 0){
                    photon68x2ContEnergy += hitE;
                }
                else if(distancePhoton < 3*ip_radius && distancePhoton > 0){
                    photon68x3ContEnergy += hitE;
                }
                else if(distancePhoton < 4*ip_radius && distancePhoton > 0){
                    photon68x4ContEnergy += hitE;
                }
                else if(distancePhoton < 5*ip_radius && distancePhoton > 0){
                    photon68x5ContEnergy += hitE;
                }

                if(distanceEle < ir_radius && distancePhoton < ip_radius && distancePhoton > 0){
                    overlap68ContEnergy += hitE;
                    overlap68Totals.at(layer) += hitE;
                }
                if(distanceEle < 2*ir_radius && distancePhoton < 2*ip_radius && distancePhoton > 0){
                    overlap68x2ContEnergy += hitE;
                }
                if(distanceEle < 3*ir_radius && distancePhoton < 3*ip_radius && distancePhoton > 0){
                    overlap68x3ContEnergy += hitE;
                }
                if(distanceEle < 4*ir_radius && distancePhoton < 4*ip_radius && distancePhoton > 0){
                    overlap68x4ContEnergy += hitE;
                }
                if(distanceEle < 5*ir_radius && distancePhoton < 5*ip_radius && distancePhoton > 0){
                    overlap68x5ContEnergy += hitE;
                }

                //double dlayer = (double)layer;

                std::vector<double> hitPosition{hitX,hitY,(double)layer,hitE};

                if(distanceEle > ir_radius && distancePhoton > ip_radius){
                    outside68Totals.at(layer) += hitE;
                    outside68NHits.at(layer) += 1;
                    outside68Xmean.at(layer) += hitX*hitE;
                    outside68Ymean.at(layer) += hitY*hitE;
                    outside68ContEnergy += hitE;
                    outside68ContNHits += 1;
                    outside68ContXmean += hitX*hitE;
                    outside68ContYmean += hitY*hitE;
                    outside68WgtCentroidCoordsX += hitX*hitE;
                    outside68WgtCentroidCoordsY += hitY*hitE;
                    //outside68HitPositions.push_back(hitPosition);           
                    outside68HitPositionsX.push_back(hitX);           
                    outside68HitPositionsY.push_back(hitY);           
                    outside68HitPositionsLayer.push_back(layer);           
                    outside68HitPositionsE.push_back(hitE);           
                }
                if(distanceEle > 2*ir_radius && distancePhoton > 2*ip_radius){
                    outside68x2ContEnergy += hitE;
                    outside68x2ContNHits += 1;
                    outside68x2ContXmean += hitX*hitE;
                    outside68x2ContYmean += hitY*hitE;
                    outside68x2WgtCentroidCoordsX += hitX*hitE;
                    outside68x2WgtCentroidCoordsY += hitY*hitE;
                    //outside68x2HitPositions.push_back(hitPosition);           
                    outside68x2HitPositionsX.push_back(hitX);           
                    outside68x2HitPositionsY.push_back(hitY);           
                    outside68x2HitPositionsLayer.push_back(layer);           
                    outside68x2HitPositionsE.push_back(hitE);           
                }
                if(distanceEle > 3*ir_radius && distancePhoton > 3*ip_radius){
                    outside68x3ContEnergy += hitE;
                    outside68x3ContNHits += 1;
                    outside68x3ContXmean += hitX*hitE;
                    outside68x3ContYmean += hitY*hitE;
                    outside68x3WgtCentroidCoordsX += hitX*hitE;
                    outside68x3WgtCentroidCoordsY += hitY*hitE;
                    //outside68x3HitPositions.push_back(hitPosition);           
                    outside68x3HitPositionsX.push_back(hitX);           
                    outside68x3HitPositionsY.push_back(hitY);           
                    outside68x3HitPositionsLayer.push_back(layer);           
                    outside68x3HitPositionsE.push_back(hitE);           
                }
                if(distanceEle > 4*ir_radius && distancePhoton > 4*ip_radius){
                    outside68x4ContEnergy += hitE;
                    outside68x4ContNHits += 1;
                    outside68x4ContXmean += hitX*hitE;
                    outside68x4ContYmean += hitY*hitE;
                    outside68x4WgtCentroidCoordsX += hitX*hitE;
                    outside68x4WgtCentroidCoordsY += hitY*hitE;
                    //outside68x4HitPositions.push_back(hitPosition);           
                    outside68x4HitPositionsX.push_back(hitX);           
                    outside68x4HitPositionsY.push_back(hitY);           
                    outside68x4HitPositionsLayer.push_back(layer);           
                    outside68x4HitPositionsE.push_back(hitE);           
                }
                if(distanceEle > 5*ir_radius && distancePhoton > 5*ip_radius){
                    outside68x5ContEnergy += hitE;
                    outside68x5ContNHits += 1;
                    outside68x5ContXmean += hitX*hitE;
                    outside68x5ContYmean += hitY*hitE;
                    outside68x5WgtCentroidCoordsX += hitX*hitE;
                    outside68x5WgtCentroidCoordsY += hitY*hitE;
                    //outside68x5HitPositions.push_back(hitPosition);           
                    outside68x5HitPositionsX.push_back(hitX);           
                    outside68x5HitPositionsY.push_back(hitY);           
                    outside68x5HitPositionsLayer.push_back(layer);           
                    outside68x5HitPositionsE.push_back(hitE);           
                }
            }

        }

        // Section ECal Hit Loop Without Found Electron Trajectory
        //This isn't used if noise is not introduced 
        if(!foundElectron && (noise || reconstruct)){
        //if(!foundElectron){
                for(int k=0; k < ecalHits->GetEntriesFast(); k++){
                    ldmx::EcalHit* hit = (ldmx::EcalHit*)ecalHits->At(k);
                    
                    double hitE = hit->getEnergy();
                    int rawID = hit->getID();

                    //The raw ID is packed with:
                    //4 bits for the subdetector, 8 cell layer bits, 3 module bits and 17 cell bits
                    //On the form: ccccccccccccccccc mmm llllllll ssss
                    //int layermask = 0xFF << 20;
                    //int layer = (layermask & rawID) >> 20;
                    int layermask = 0xFF << 4;
                    int layer = (layermask & rawID) >> 4;            

                    //int modulemask = 0x7 << 17;
                    //int module = (modulemask & rawID) >> 17;
                    int modulemask = 0x7 << 12;
                    int module = (modulemask & rawID) >> 12;

                    //int cellmask = 0x1FFFF;
                    //int cell = rawID & 0x1FFFF;
                    int cellmask = 0x1FFFF << 15;
                    int cell = (rawID & cellmask) >> 15;

                    int mcid_val = 10*cell+module;
                    //int mcid_index_slow = findCellIndex(mcid, mcid_val);
                    int mcid_index = (mcid_val % 10)*396 + (mcid_val - (mcid_val % 10))/10+ (mcid_val % 10);// + 1;
                   // std::cout << mcid_val << ":" <<mcid_index_slow << ":" << mcid_index << std::endl;

                    double hitX = cellX.at(mcid_index);
                    double hitY = cellY.at(mcid_index);
                    
                    allE.push_back(hitE);
                    allX.push_back(hitX);
                    allY.push_back(hitY);
                    allLayer.push_back(layer);
                    allmcid.push_back(mcid_index);

                    occupiedCellsProjected[mcid_index] = true;
                    occupiedCells[layer*2779+mcid_index] = true;


                    if(noise){ 
                        if(hit->isNoise()){
                            continue;
                        }
                    }
                    if(noise){
                        hitE += rng->Gaus(0,noiseFactor*1506.32*0.13/33000);
     
                        if(!(hitE > 4*noiseFactor*1506.32*0.13/33000)){
                            continue;
                        }
                    }
                    else{
                        if(!(hitE > 0)){
                            continue;
                        }
                    }

                    //allE.push_back(hitE);
                    //allX.push_back(hitX);
                    //allY.push_back(hitY);
                    //allLayer.push_back(layer);
                    //allmcid.push_back(mcid_index);

                    //occupiedCellsProjected[mcid_index] = true;
                    //occupiedCells[layer*2779+mcid_index] = true;

                    wgtCentroidCoordsX += hitX*hitE;
                    wgtCentroidCoordsY += hitY*hitE;
                    
                    nHits += 1;
                    summedDet += hitE;
                    if(layer < 20){
                        frontEcal += hitE;
                    }

                    xMean += hitX*hitE;
                    yMean += hitY*hitE;
                    
                    avgLayerHit += layer;
                    wavgLayerHit += layer*hitE;

                }
        }

        // Section Additional Noise Generation
        /*
        Noise generation on empty hits

        choose noiseHits of empty cells and add Gaussian noise to them
        where the noise is beyond the 4th sigma
        */
        
        //Hardcoded with sigma = 1.25 of original noise
        int newNoiseHits = rng->Binomial(2779*34, noiseProb);

        if(noise){
            for(int k = 0; k < newNoiseHits; k++){
                while(true){
                    Int_t place = rng->Integer(2779*34);
                    if(occupiedCells[place] == false){
        
                        Int_t mcid_index = place % 2779;

                        Int_t layer = (Int_t)((place - (place % 2779))/2779);

                        //Add hit with 1.25*4ms
                        double hitE = sample4SigmaTail(rng)*noiseFactor*1506.32*0.13/33000;
                        occupiedCells[place] = true;
                        occupiedCellsProjected[mcid_index] = true;
                
                        summedDet += hitE;
                        nHits++;
                        if(layer < 20){
                            frontEcal += hitE;
                        }
                        
                        double hitX = cellX.at(mcid_index);
                        double hitY = cellY.at(mcid_index);
                        
                        allE.push_back(hitE);
                        allX.push_back(hitX);
                        allY.push_back(hitY);
                        allLayer.push_back(layer);
                        allmcid.push_back(mcid_index);
                        
                        xMean += hitX*hitE;
                        yMean += hitY*hitE;

                        avgLayerHit += layer;
                        wavgLayerHit += layer*hitE;

                        wgtCentroidCoordsX += hitX*hitE;
                        wgtCentroidCoordsY += hitY*hitE;

                        if(foundElectron){

                            double distanceEle = TMath::Sqrt(pow(hitX -elePosX.at(layer),2)+pow(hitY-elePosY.at(layer),2));

                            //double distancePhoton = TMath::Sqrt(pow(hitX -photonPosX.at(layer),2)+pow(hitY-photonPosY.at(layer),2));
                            double distancePhoton = -1;
                            if(foundTargetElectron){
                                distancePhoton = TMath::Sqrt(pow(hitX -photonPosX.at(layer),2)+pow(hitY-photonPosY.at(layer),2));
                            }
         
                            if(layer >= 20){
                                ecalBackEnergy += hitE;
                            }

                            double ir_radius = radii.at((ir-1)*34+layer);

                            if(distanceEle < ir_radius){
                                ele68ContEnergy += hitE;
                                ele68Totals.at(layer) += hitE;
                            }
                            else if(distanceEle < 2*ir_radius){
                                ele68x2ContEnergy += hitE;
                            }
                            else if(distanceEle < 3*ir_radius){
                                ele68x3ContEnergy += hitE;
                            }
                            else if(distanceEle < 4*ir_radius){
                                ele68x4ContEnergy += hitE;
                            }
                            else if(distanceEle < 5*ir_radius){
                                ele68x5ContEnergy += hitE;
                            }
                    
                            double ip_radius = radii.at(layer);
                            
                            if(distancePhoton < ip_radius && distancePhoton > 0){
                                photon68ContEnergy += hitE;
                                photon68Totals.at(layer) += hitE;
                            }
                            else if(distancePhoton < 2*ip_radius && distancePhoton > 0){
                                photon68x2ContEnergy += hitE;
                            }
                            else if(distancePhoton < 3*ip_radius && distancePhoton > 0){
                                photon68x3ContEnergy += hitE;
                            }
                            else if(distancePhoton < 4*ip_radius && distancePhoton > 0){
                                photon68x4ContEnergy += hitE;
                            }
                            else if(distancePhoton < 5*ip_radius && distancePhoton > 0){
                                photon68x5ContEnergy += hitE;
                            }

                            if(distanceEle < ir_radius && distancePhoton < ip_radius && distancePhoton > 0){
                                overlap68ContEnergy += hitE;
                                overlap68Totals.at(layer) += hitE;
                            }
                            if(distanceEle < 2*ir_radius && distancePhoton < 2*ip_radius && distancePhoton > 0){
                                overlap68x2ContEnergy += hitE;
                            }
                            if(distanceEle < 3*ir_radius && distancePhoton < 3*ip_radius && distancePhoton > 0){
                                overlap68x3ContEnergy += hitE;
                            }
                            if(distanceEle < 4*ir_radius && distancePhoton < 4*ip_radius && distancePhoton > 0){
                                overlap68x4ContEnergy += hitE;
                            }
                            if(distanceEle < 5*ir_radius && distancePhoton < 5*ip_radius && distancePhoton > 0){
                                overlap68x5ContEnergy += hitE;
                            }

                            std::vector<double> hitPosition{hitX,hitY,(double)layer,hitE};

                            if(distanceEle > ir_radius && distancePhoton > ip_radius){
                                outside68Totals.at(layer) += hitE;
                                outside68NHits.at(layer) += 1;
                                outside68Xmean.at(layer) += hitX*hitE;
                                outside68Ymean.at(layer) += hitY*hitE;
                                outside68ContEnergy += hitE;
                                outside68ContNHits += 1;
                                outside68ContXmean += hitX*hitE;
                                outside68ContYmean += hitY*hitE;
                                outside68WgtCentroidCoordsX += hitX*hitE;
                                outside68WgtCentroidCoordsY += hitY*hitE;
                                //outside68HitPositions.push_back(hitPosition);           
                                outside68HitPositionsX.push_back(hitX);           
                                outside68HitPositionsY.push_back(hitY);           
                                outside68HitPositionsLayer.push_back(layer);           
                                outside68HitPositionsE.push_back(hitE);           
                            }
                            if(distanceEle > 2*ir_radius && distancePhoton > 2*ip_radius){
                                outside68x2ContEnergy += hitE;
                                outside68x2ContNHits += 1;
                                outside68x2ContXmean += hitX*hitE;
                                outside68x2ContYmean += hitY*hitE;
                                outside68x2WgtCentroidCoordsX += hitX*hitE;
                                outside68x2WgtCentroidCoordsY += hitY*hitE;
                                //outside68x2HitPositions.push_back(hitPosition);           
                                outside68x2HitPositionsX.push_back(hitX);           
                                outside68x2HitPositionsY.push_back(hitY);           
                                outside68x2HitPositionsLayer.push_back(layer);           
                                outside68x2HitPositionsE.push_back(hitE);           
                            }
                            if(distanceEle > 3*ir_radius && distancePhoton > 3*ip_radius){
                                outside68x3ContEnergy += hitE;
                                outside68x3ContNHits += 1;
                                outside68x3ContXmean += hitX*hitE;
                                outside68x3ContYmean += hitY*hitE;
                                outside68x3WgtCentroidCoordsX += hitX*hitE;
                                outside68x3WgtCentroidCoordsY += hitY*hitE;
                                //outside68x3HitPositions.push_back(hitPosition);           
                                outside68x3HitPositionsX.push_back(hitX);           
                                outside68x3HitPositionsY.push_back(hitY);           
                                outside68x3HitPositionsLayer.push_back(layer);           
                                outside68x3HitPositionsE.push_back(hitE);           
                            }
                            if(distanceEle > 4*ir_radius && distancePhoton > 4*ip_radius){
                                outside68x4ContEnergy += hitE;
                                outside68x4ContNHits += 1;
                                outside68x4ContXmean += hitX*hitE;
                                outside68x4ContYmean += hitY*hitE;
                                outside68x4WgtCentroidCoordsX += hitX*hitE;
                                outside68x4WgtCentroidCoordsY += hitY*hitE;
                                //outside68x4HitPositions.push_back(hitPosition);           
                                outside68x4HitPositionsX.push_back(hitX);           
                                outside68x4HitPositionsY.push_back(hitY);           
                                outside68x4HitPositionsLayer.push_back(layer);           
                                outside68x4HitPositionsE.push_back(hitE);           
                            }
                            if(distanceEle > 5*ir_radius && distancePhoton > 5*ip_radius){
                                outside68x5ContEnergy += hitE;
                                outside68x5ContNHits += 1;
                                outside68x5ContXmean += hitX*hitE;
                                outside68x5ContYmean += hitY*hitE;
                                outside68x5WgtCentroidCoordsX += hitX*hitE;
                                outside68x5WgtCentroidCoordsY += hitY*hitE;
                                //outside68x5HitPositions.push_back(hitPosition);           
                                outside68x5HitPositionsX.push_back(hitX);           
                                outside68x5HitPositionsY.push_back(hitY);           
                                outside68x5HitPositionsLayer.push_back(layer);           
                                outside68x5HitPositionsE.push_back(hitE);           
                            }
                        }
                        break;
                    }
                }
            }
        }

        // Section Containment Variable Calculations

        double outside68ContShowerRMS = 0;
        outside68ContXstd = 0;       
        outside68ContYstd = 0;       
        std::vector<double> outside68Xstd{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        std::vector<double> outside68Ystd{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        //for(auto hit = outside68HitPositions.begin(); hit < outside68HitPositions.end(); hit++){
        for(int k = 0; k < outside68HitPositionsX.size(); k++){
            //double distanceCentroid = TMath::Sqrt(pow(hit->at(0)-outside68WgtCentroidCoordsX,2)+pow(hit->at(1)-outside68WgtCentroidCoordsY,2));
            double distanceCentroid = TMath::Sqrt(pow(outside68HitPositionsX.at(k)-outside68WgtCentroidCoordsX,2)+pow(outside68HitPositionsY.at(k)-outside68WgtCentroidCoordsY,2));

            outside68ContShowerRMS += distanceCentroid*(outside68HitPositionsE.at(k));
            //outside68ContShowerRMS += distanceCentroid*(hit->at(3));

            //Converted back from float, hope it's okay
            //int layer = (int)hit->at(2);
            int layer = (int)outside68HitPositionsLayer.at(k);

            if(outside68Totals.at(layer) > 0){
                //outside68Xstd.at(layer) += pow(hit->at(0)-outside68Xmean.at(layer)/outside68Totals.at(layer),2)*(hit->at(3));
                outside68Xstd.at(layer) += pow(outside68HitPositionsX.at(k)-outside68Xmean.at(layer)/outside68Totals.at(layer),2)*(outside68HitPositionsE.at(k));
                //outside68Ystd.at(layer) += pow(hit->at(1)-outside68Ymean.at(layer)/outside68Totals.at(layer),2)*(hit->at(3));
                outside68Ystd.at(layer) += pow(outside68HitPositionsY.at(k)-outside68Ymean.at(layer)/outside68Totals.at(layer),2)*(outside68HitPositionsE.at(k));
            }
            if(outside68ContEnergy > 0){
                //outside68ContXstd += pow(hit->at(0)-outside68ContXmean/outside68ContEnergy,2)*(hit->at(3));
                outside68ContXstd += pow(outside68HitPositionsX.at(k)-outside68ContXmean/outside68ContEnergy,2)*(outside68HitPositionsE.at(k));
                //outside68ContYstd += pow(hit->at(1)-outside68ContYmean/outside68ContEnergy,2)*(hit->at(3));
                outside68ContYstd += pow(outside68HitPositionsY.at(k)-outside68ContYmean/outside68ContEnergy,2)*(outside68HitPositionsE.at(k));
            }

        }
 
        double outside68x2ContShowerRMS = 0;
        outside68x2ContXstd = 0;       
        outside68x2ContYstd = 0;
        
        //for(auto hit = outside68x2HitPositions.begin(); hit < outside68x2HitPositions.end(); hit++){
        for(int k = 0; k < outside68x2HitPositionsX.size(); k++){
            double distanceCentroid = TMath::Sqrt(pow(outside68x2HitPositionsX.at(k)-outside68x2WgtCentroidCoordsX,2)+pow(outside68x2HitPositionsY.at(k)-outside68x2WgtCentroidCoordsY,2));

            outside68x2ContShowerRMS += distanceCentroid*(outside68x2HitPositionsE.at(k));

            if(outside68x2ContEnergy > 0){
                outside68x2ContXstd += pow(outside68x2HitPositionsX.at(k)-outside68x2ContXmean/outside68x2ContEnergy,2)*(outside68x2HitPositionsE.at(k));
                outside68x2ContYstd += pow(outside68x2HitPositionsY.at(k)-outside68x2ContYmean/outside68x2ContEnergy,2)*(outside68x2HitPositionsE.at(k));
            }
        }
        
        double outside68x3ContShowerRMS = 0;
        outside68x3ContXstd = 0;       
        outside68x3ContYstd = 0;
        
        //for(auto hit = outside68x3HitPositions.begin(); hit < outside68x3HitPositions.end(); hit++){
        for(int k = 0; k < outside68x3HitPositionsX.size(); k++){
            //double distanceCentroid = TMath::Sqrt(pow(hit->at(0)-outside68x3WgtCentroidCoordsX,2)+pow(hit->at(1)-outside68x3WgtCentroidCoordsY,2));
            double distanceCentroid = TMath::Sqrt(pow(outside68x3HitPositionsX.at(k)-outside68x3WgtCentroidCoordsX,2)+pow(outside68x3HitPositionsY.at(k)-outside68x3WgtCentroidCoordsY,2));

            outside68x3ContShowerRMS += distanceCentroid*(outside68x3HitPositionsE.at(k));

            if(outside68x3ContEnergy > 0){
                outside68x3ContXstd += pow(outside68x3HitPositionsX.at(k)-outside68x3ContXmean/outside68x3ContEnergy,2)*(outside68x3HitPositionsE.at(k));
                outside68x3ContYstd += pow(outside68x3HitPositionsY.at(k)-outside68x3ContYmean/outside68x3ContEnergy,2)*(outside68x3HitPositionsE.at(k));
            }
        }
        
        double outside68x4ContShowerRMS = 0;
        outside68x4ContXstd = 0;       
        outside68x4ContYstd = 0;
        
        //for(auto hit = outside68x4HitPositions.begin(); hit < outside68x4HitPositions.end(); hit++){
        for(int k = 0; k < outside68x4HitPositionsX.size(); k++){
            double distanceCentroid = TMath::Sqrt(pow(outside68x4HitPositionsX.at(k)-outside68x4WgtCentroidCoordsX,2)+pow(outside68x4HitPositionsY.at(k)-outside68x4WgtCentroidCoordsY,2));

            outside68x4ContShowerRMS += distanceCentroid*(outside68x4HitPositionsE.at(k));

            if(outside68x4ContEnergy > 0){
                outside68x4ContXstd += pow(outside68x4HitPositionsX.at(k)-outside68x4ContXmean/outside68x4ContEnergy,2)*(outside68x4HitPositionsE.at(k));
                outside68x4ContYstd += pow(outside68x4HitPositionsY.at(k)-outside68x4ContYmean/outside68x4ContEnergy,2)*(outside68x4HitPositionsE.at(k));
            }
        }
        
        double outside68x5ContShowerRMS = 0;
        outside68x5ContXstd = 0;       
        outside68x5ContYstd = 0;
        
        //for(auto hit = outside68x5HitPositions.begin(); hit < outside68x5HitPositions.end(); hit++){
        for(int k = 0; k < outside68x5HitPositionsX.size(); k++){
            double distanceCentroid = TMath::Sqrt(pow(outside68x5HitPositionsX.at(k)-outside68x5WgtCentroidCoordsX,2)+pow(outside68x5HitPositionsY.at(k)-outside68x5WgtCentroidCoordsY,2));

            outside68x5ContShowerRMS += distanceCentroid*(outside68x5HitPositionsE.at(k));

            if(outside68x5ContEnergy > 0){
                outside68x5ContXstd += pow(outside68x5HitPositionsX.at(k)-outside68x5ContXmean/outside68x5ContEnergy,2)*(outside68x5HitPositionsE.at(k));
                outside68x5ContYstd += pow(outside68x5HitPositionsY.at(k)-outside68x5ContYmean/outside68x5ContEnergy,2)*(outside68x5HitPositionsE.at(k));
            }
        }

        if(outside68ContEnergy > 0){
            outside68ContXmean /= outside68ContEnergy;
            outside68ContYmean /= outside68ContEnergy;
            outside68ContXstd = TMath::Sqrt(outside68ContXstd/outside68ContEnergy);
            outside68ContYstd = TMath::Sqrt(outside68ContYstd/outside68ContEnergy);
        }
        if(outside68x2ContEnergy > 0){
            outside68x2ContXmean /= outside68x2ContEnergy;
            outside68x2ContYmean /= outside68x2ContEnergy;
            outside68x2ContXstd = TMath::Sqrt(outside68x2ContXstd/outside68x2ContEnergy);
            outside68x2ContYstd = TMath::Sqrt(outside68x2ContYstd/outside68x2ContEnergy);
        }
        if(outside68x3ContEnergy > 0){
            outside68x3ContXmean /= outside68x3ContEnergy;
            outside68x3ContYmean /= outside68x3ContEnergy;
            outside68x3ContXstd = TMath::Sqrt(outside68x3ContXstd/outside68x3ContEnergy);
            outside68x3ContYstd = TMath::Sqrt(outside68x3ContYstd/outside68x3ContEnergy);
        }
        if(outside68x4ContEnergy > 0){
            outside68x4ContXmean /= outside68x4ContEnergy;
            outside68x4ContYmean /= outside68x4ContEnergy;
            outside68x4ContXstd = TMath::Sqrt(outside68x4ContXstd/outside68x4ContEnergy);
            outside68x4ContYstd = TMath::Sqrt(outside68x4ContYstd/outside68x4ContEnergy);
        }
        if(outside68x5ContEnergy > 0){
            outside68x5ContXmean /= outside68x5ContEnergy;
            outside68x5ContYmean /= outside68x5ContEnergy;
            outside68x5ContXstd = TMath::Sqrt(outside68x5ContXstd/outside68x5ContEnergy);
            outside68x5ContYstd = TMath::Sqrt(outside68x5ContYstd/outside68x5ContEnergy);
        }
       
        // Section Other BDT Variable Calculations
 
        //Recalculate all other BDT features if noise is added
        if(noise || reconstruct){
            if(nHits > 0){
                xMean /= summedDet;
                yMean /= summedDet;
                avgLayerHit /= nHits;
                wavgLayerHit /= summedDet;
            }
            else{
                xMean = 0;
                yMean = 0;
                avgLayerHit = 0;
                wavgLayerHit = 0;
            }

            for(int k = 0; k < allE.size(); k++){
                xStd += pow(allX.at(k)-xMean,2)*allE.at(k);
                yStd += pow(allY.at(k)-yMean,2)*allE.at(k);
                stdLayerHit += pow(allLayer.at(k)-wavgLayerHit,2)*allE.at(k);
                
                if(allLayer.at(k) > deepestLayerHit){
                    if(allE.at(k) > 0 ){
                        deepestLayerHit = allLayer.at(k);
                    }
                }
                if(allE.at(k) > maxCellDep){
                    maxCellDep = allE.at(k);
                }
            }
            if(nHits > 0){
                xStd = TMath::Sqrt(xStd / summedDet);
                yStd = TMath::Sqrt(yStd / summedDet);
                stdLayerHit = TMath::Sqrt(stdLayerHit/summedDet);
            }
            else{
                xStd = 0;
                yStd = 0;
                stdLayerHit = 0;
            }

            wgtCentroidCoordsX = (summedDet > 1E-6) ? wgtCentroidCoordsX / summedDet : wgtCentroidCoordsX;
            wgtCentroidCoordsY = (summedDet > 1E-6) ? wgtCentroidCoordsY / summedDet : wgtCentroidCoordsY;

            int centroidIndex = -1;
            //double showerRMS = 0;
            double maxDist = 1e6;

            for(int k = 0; k < allE.size(); k++){
                double deltaR = pow(pow((allX.at(k)-wgtCentroidCoordsX),2)+pow((allY.at(k)-wgtCentroidCoordsY),2),0.5);
                showerRMS += deltaR*allE.at(k);
                if(deltaR < maxDist){
                    maxDist = deltaR;
                    centroidIndex = k;
                }
            }
            
            if(centroidIndex >= 0){
                int centroidMcid = allmcid.at(centroidIndex);


                if(summedDet > 0){
                    showerRMS = showerRMS / summedDet;
                }

                //double summedTightIso = 0;

                for(int k = 0; k < allmcid.size(); k++){

                    //Discard hit on centroid
                    if(allmcid.at(k) == centroidMcid){
                        continue;
                    }
                    
                    bool NNCollision = false;

                    //Discard nearest neighbours to centroid hit
                    //for(int j = 0; j < NNMap[allmcid.at(k)].size(); j++){
                    for(int j = 0; j < NNMap[centroidMcid].size(); j++){
                        if(NNMap[centroidMcid].at(j) == allmcid.at(k)){
                            NNCollision = true;
                        }
                    }
                    if(NNCollision){
                        continue;
                    }

                    //Discard non-isolated hits
                    for(int j = 0; j < NNMap[allmcid.at(k)].size(); j++){
                        //std::cout << allLayer.at(k) << ":" << NNMap[allmcid.at(k)].at(j) << std::endl;
                        int n = allLayer.at(k)*2779+(int)NNMap[allmcid.at(k)].at(j); 
                        //if(occupiedCells[allLayer.at(k)*2779+(int)NNMap[allmcid.at(k)].at(j)] == true){ 
                        if(occupiedCells[n] == true){ 
                        //if(occupiedCells[NNMap[allmcid.at(k)].at(j)] == true){ 
                            NNCollision = true; 
                        }
                    }
                    if(NNCollision){
                        continue;
                    }

                    summedTightIso += allE.at(k);

                }
            }
        }
        else{ //Otherwise if you are not reconstructing, just take them from the input ROOT file
            //TPython::Exec("intree.GetEntry(i); i +=1");
            nHits = TPython::Eval("ecalVetoRes[0].getNReadoutHits()");
            summedDet = TPython::Eval("ecalVetoRes[0].getSummedDet()");
            summedTightIso = TPython::Eval("ecalVetoRes[0].getSummedTightIso()");
            maxCellDep = TPython::Eval("ecalVetoRes[0].getMaxCellDep()");
            showerRMS = TPython::Eval("ecalVetoRes[0].getShowerRMS()");
            xStd = TPython::Eval("ecalVetoRes[0].getXStd()");
            yStd = TPython::Eval("ecalVetoRes[0].getYStd()");
            avgLayerHit = TPython::Eval("ecalVetoRes[0].getAvgLayerHit()");
            stdLayerHit = TPython::Eval("ecalVetoRes[0].getStdLayerHit()");
            deepestLayerHit = TPython::Eval("ecalVetoRes[0].getDeepestLayerHit()");
        }
        score = -1.0;

        // Section Evaluate BDT
               
        std::vector<double> bdtFeatures{(double)nHits, (double)summedDet,(double)summedTightIso,maxCellDep,(double)showerRMS,(double)xStd,(double)yStd,(double)avgLayerHit,(double)deepestLayerHit,(double)stdLayerHit,ele68ContEnergy,ele68x2ContEnergy,ele68x3ContEnergy,ele68x4ContEnergy,ele68x5ContEnergy,photon68ContEnergy,photon68x2ContEnergy,photon68x3ContEnergy,photon68x4ContEnergy,photon68x5ContEnergy,outside68ContEnergy,outside68x2ContEnergy,outside68x3ContEnergy,outside68x4ContEnergy,outside68x5ContEnergy,outside68ContNHits,outside68x2ContNHits,outside68x3ContNHits,outside68x4ContNHits,outside68x5ContNHits,outside68ContXstd,outside68x2ContXstd,outside68x3ContXstd,outside68x4ContXstd,outside68x5ContXstd,outside68ContYstd,outside68x2ContYstd,outside68x3ContYstd,outside68x4ContYstd,outside68x5ContYstd,(double)ecalBackEnergy};
 
        TString cmd = vectorToPredCMD(bdtFeatures);

        score = TPython::Eval(cmd);
 
        /*
        std::cout << "ecalBackEnergy: " << ecalBackEnergy<< std::endl;
        std::cout << "outside68x5ContYStd: " << outside68x5ContYstd<< std::endl;
        std::cout << "outside68x4ContYStd: " << outside68x4ContYstd<< std::endl;
        std::cout << "outside68x3ContYStd: " << outside68x3ContYstd<< std::endl;
        std::cout << "outside68x2ContYStd: " << outside68x2ContYstd<< std::endl;
        std::cout << "outside68ContYStd: " << outside68ContYstd<< std::endl;
        std::cout << "outside68x5ContXStd: " << outside68x5ContXstd<< std::endl;
        std::cout << "outside68x4ContXStd: " << outside68x4ContXstd<< std::endl;
        std::cout << "outside68x3ContXStd: " << outside68x3ContXstd<< std::endl;
        std::cout << "outside68x2ContXStd: " << outside68x2ContXstd<< std::endl;
        std::cout << "outside68ContXStd: " << outside68ContXstd<< std::endl;
        std::cout << "outside68x5ContNHits: " << outside68x5ContNHits<< std::endl;
        std::cout << "outside68x4ContNHits: " << outside68x4ContNHits<< std::endl;
        std::cout << "outside68x3ContNHits: " << outside68x3ContNHits<< std::endl;
        std::cout << "outside68x2ContNHits: " << outside68x2ContNHits<< std::endl;
        std::cout << "outside68ContNHits: " << outside68ContNHits<< std::endl;
        std::cout << "outside68x5ContEnergy: " << outside68x5ContEnergy<< std::endl;
        std::cout << "outside68x4ContEnergy: " << outside68x4ContEnergy<< std::endl;
        std::cout << "outside68x3ContEnergy: " << outside68x3ContEnergy<< std::endl;
        std::cout << "outside68x2ContEnergy: " << outside68x2ContEnergy<< std::endl;
        std::cout << "outside68ContEnergy: " << outside68ContEnergy<< std::endl;
        std::cout << "photon68x5ContEnergy: " << photon68x5ContEnergy<< std::endl;
        std::cout << "photon68x4ContEnergy: " << photon68x4ContEnergy<< std::endl;
        std::cout << "photon68x3ContEnergy: " << photon68x3ContEnergy<< std::endl;
        std::cout << "photon68x2ContEnergy: " << photon68x2ContEnergy<< std::endl;
        std::cout << "photon68ContEnergy: " << photon68ContEnergy<< std::endl;
        std::cout << "ele68x5ContEnergy: " << ele68x5ContEnergy<< std::endl;
        std::cout << "ele68x4ContEnergy: " << ele68x4ContEnergy<< std::endl;
        std::cout << "ele68x3ContEnergy: " << ele68x3ContEnergy<< std::endl;
        std::cout << "ele68x2ContEnergy: " << ele68x2ContEnergy<< std::endl;
        std::cout << "ele68ContEnergy: " << ele68ContEnergy<< std::endl;
        std::cout << "stdLayerHit: " << stdLayerHit << std::endl;
        std::cout << "deepestLayerHit: " << deepestLayerHit<< std::endl;
        std::cout << "avgLayerHit: " << avgLayerHit<< std::endl;
        std::cout << "yStd: " << yStd<< std::endl;
        std::cout << "xStd: " << xStd<< std::endl;
        std::cout << "showerRMS: " << showerRMS<< std::endl;
        std::cout << "maxCellDep: " << maxCellDep<< std::endl;
        std::cout << "summedTightIso: " << summedTightIso<< std::endl;
        std::cout << "summedDet: " << summedDet<< std::endl;
        std::cout << "nHits: " << nHits<< std::endl;

        std::cout << "Score: " << score << std::endl;
        */   
 
        if(score > 0.999){
            passed0_999 += 1;
        }       

        if(score > 0.999 && passesHcalVeto && passesTrackVeto){
            passedAll += 1;
        }

        //Save everything to the output file
        outtree->Fill();

    }

    std::cout << "Passed track veto: " << passedTrack << std::endl;
    std::cout << "Passed Hcal veto: " << passedHcal << std::endl;
    std::cout << "Passed 0.999 BDT score: " << passed0_999 << std::endl;
    std::cout << "Passed all three cuts: " << passedAll << std::endl;

    std::cout << "Closing files" << std::endl;

    outtree->Write();
    outfile->Close();
    infile->Close();
}

