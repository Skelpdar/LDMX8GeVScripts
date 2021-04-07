/*
 *Evaluates the BDT score of ldmx-sw 1.7.0 data with the v9 geometry
 *Also forms a skeleton for any other analysis needing some re-reconstruction
 *
 *Erik Wallin
 *
 *Adapted from EventProc/src/EcalVetoProcessor.cxx in ldmx-sw 1.7.0
 *and
 *EcalVeto/bdtTreeMaker.py in IncandelaLab/LDMX-scripts
 */

R__LOAD_LIBRARY(/PathToSharedLibraries/libEvent.so)

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TPython.h"
#include "TObject.h"
#include "TMath.h"

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

//Path to the header files must be specified before this macro starts.
//E.g. with gROOT->ProcessLine(".include PATH/Event/include");
//in rootlogon.C in the same directory

#include "Event/EcalHit.h"
#include "Event/CalorimeterHit.h"
#include "Event/SimParticle.h"
#include "Event/SimTrackerHit.h"
#include "Event/EcalVetoResult.h"

//Code snippet from ldmx-sw
//Translates a vector of doubles into line of Python code that evaluates the BDT score
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

int findCellIndex(std::vector<int> ids, int cellid){
    //2780 channels
    for(int i=0; i < 2779; i++){
        if(ids.at(i) == cellid){
            return i;
        }
    }
    return -1;
}

void eval(TString infilename, TString outfilename){
    std::cout << "Starting BDT evaluation" << std::endl;

    TPython::Exec("print(\'Loading xgboost BDT\')"); 
    TPython::Exec("import xgboost as xgb");
    TPython::Exec("import numpy as np");
    TPython::Exec("import pickle as pkl");
    TPython::Exec("print(\"xgboost version: \" + xgb.__version__)");
    TPython::Exec("model = pkl.load(open(\'16001439_0_weights.pkl\','rb'))");

    TPython::Exec("print(\"Testing loaded model with zeros input\")");
    TPython::Exec("pred = float(model.predict(xgb.DMatrix(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))[0])");

    TPython::Exec("print(\"Test score: \" + str(pred))");

    //Load cell map
    TPython::Exec("cellMap = np.loadtxt(\'cellmodule.txt\')");
    std::vector<int> mcid;
    std::vector<float> cellX;
    std::vector<float> cellY;

    //Lazily run the loop inside Python to bother with string concatenation
    TPython::Exec("i_ = 0");
    for(int i=0; i < 2779; i++){
        mcid.push_back(TPython::Eval("cellMap[i_,0]"));
        cellX.push_back(TPython::Eval("cellMap[i_,1]"));
        cellY.push_back(TPython::Eval("cellMap[i_,2]"));

        TPython::Exec("i_ += 1");
    }
    
    //Constants

    std::vector<double> layerZs{223.8000030517578, 226.6999969482422, 233.0500030517578, 237.4499969482422, 245.3000030517578, 251.1999969482422, 260.29998779296875,266.70001220703125, 275.79998779296875, 282.20001220703125, 291.29998779296875, 297.70001220703125, 306.79998779296875, 313.20001220703125,322.29998779296875, 328.70001220703125, 337.79998779296875, 344.20001220703125, 353.29998779296875, 359.70001220703125, 368.79998779296875,375.20001220703125, 384.29998779296875, 390.70001220703125, 403.29998779296875, 413.20001220703125, 425.79998779296875, 435.70001220703125,448.29998779296875, 458.20001220703125, 470.79998779296875, 480.70001220703125, 493.29998779296875, 503.20001220703125};

    std::vector<double> radius_recoil_68_p_0_1000_theta_0_6{9.65155163169527,9.029949540117707,8.169116380219359,7.26878332423302,5.723387467629167,5.190678018534044,5.927290663506518,6.182560329200212,7.907549398117859,8.606100542857211,10.93381822596916,12.043201938160239,14.784548371508041,16.102403056546482,18.986402399412817,20.224453740305716,23.048820910305643,24.11202594672678,26.765135236851666,27.78700483852502,30.291794353801293,31.409870873194464,33.91006482486666,35.173073672355926,38.172422630271,40.880288341493205,44.696485719120005,49.23802839743545,53.789910813378675,60.87843355562641,66.32931132415688,75.78117972604727,86.04697356716805,96.90360704034346};
    std::vector<double> radius_recoil_68_p_1000_end_theta_0_6{9.65155163169527,9.029949540117707,8.169116380219359,7.26878332423302,5.723387467629167,5.190678018534044,5.927290663506518,6.182560329200212,7.907549398117859,8.606100542857211,10.93381822596916,12.043201938160239,14.784548371508041,16.102403056546482,18.986402399412817,20.224453740305716,23.048820910305643,24.11202594672678,26.765135236851666,27.78700483852502,30.291794353801293,31.409870873194464,33.91006482486666,35.173073672355926,38.172422630271,40.880288341493205,44.696485719120005,49.23802839743545,53.789910813378675,60.87843355562641,66.32931132415688,75.78117972604727,86.04697356716805,96.90360704034346};
    std::vector<double> radius_recoil_68_theta_6_15{6.486368894455214,6.235126063894043,6.614742647173138,7.054111110170857,7.6208431229479645,8.262931570498493,10.095697703256274,11.12664183734125,13.463274649564859,14.693527904936063,17.185557959405358,18.533873226278285,21.171912124279075,22.487821335146958,25.27214729142235,26.692900194943586,29.48033347163334,30.931911179461117,33.69749728369263,35.35355537189422,37.92163028706617,40.08541101327325,42.50547781670488,44.42600915526537,48.18838292957783,50.600428280254235,55.85472906972822,60.88022977643599,68.53506382625108,73.0547148939902,78.01129860152466,90.91421661272666,104.54696678290463,116.90671501444335};
    std::vector<double> radius_recoil_68_theta_15_end{7.218181823299591,7.242577749118457,9.816977116964644,12.724324104744532,17.108322705113288,20.584866353828193,25.036863838363544,27.753201816619153,32.08174405069556,34.86092888550297,39.56748303616661,43.37808998888681,48.50525488266305,52.66203291220487,58.00763047516536,63.028585648616584,69.21745026096245,74.71857224945907,82.15269906028466,89.1198060894434,95.15548897621329,103.91086738998598,106.92403611582472,115.76216727231979,125.72534759956525,128.95688953061537,140.84273174274335,151.13069543119798,163.87399183389545,171.8032189173357,186.89216628021853,200.19270470457505,219.32987417488016,236.3947885046377};


    //Just the containment radii vectors concatenated
    std::vector<double> radii{9.65155163169527,9.029949540117707,8.169116380219359,7.26878332423302,5.723387467629167,5.190678018534044,5.927290663506518,6.182560329200212,7.907549398117859,8.606100542857211,10.93381822596916,12.043201938160239,14.784548371508041,16.102403056546482,18.986402399412817,20.224453740305716,23.048820910305643,24.11202594672678,26.765135236851666,27.78700483852502,30.291794353801293,31.409870873194464,33.91006482486666,35.173073672355926,38.172422630271,40.880288341493205,44.696485719120005,49.23802839743545,53.789910813378675,60.87843355562641,66.32931132415688,75.78117972604727,86.04697356716805,96.90360704034346,9.65155163169527,9.029949540117707,8.169116380219359,7.26878332423302,5.723387467629167,5.190678018534044,5.927290663506518,6.182560329200212,7.907549398117859,8.606100542857211,10.93381822596916,12.043201938160239,14.784548371508041,16.102403056546482,18.986402399412817,20.224453740305716,23.048820910305643,24.11202594672678,26.765135236851666,27.78700483852502,30.291794353801293,31.409870873194464,33.91006482486666,35.173073672355926,38.172422630271,40.880288341493205,44.696485719120005,49.23802839743545,53.789910813378675,60.87843355562641,66.32931132415688,75.78117972604727,86.04697356716805,96.90360704034346,6.486368894455214,6.235126063894043,6.614742647173138,7.054111110170857,7.6208431229479645,8.262931570498493,10.095697703256274,11.12664183734125,13.463274649564859,14.693527904936063,17.185557959405358,18.533873226278285,21.171912124279075,22.487821335146958,25.27214729142235,26.692900194943586,29.48033347163334,30.931911179461117,33.69749728369263,35.35355537189422,37.92163028706617,40.08541101327325,42.50547781670488,44.42600915526537,48.18838292957783,50.600428280254235,55.85472906972822,60.88022977643599,68.53506382625108,73.0547148939902,78.01129860152466,90.91421661272666,104.54696678290463,116.90671501444335,7.218181823299591,7.242577749118457,9.816977116964644,12.724324104744532,17.108322705113288,20.584866353828193,25.036863838363544,27.753201816619153,32.08174405069556,34.86092888550297,39.56748303616661,43.37808998888681,48.50525488266305,52.66203291220487,58.00763047516536,63.028585648616584,69.21745026096245,74.71857224945907,82.15269906028466,89.1198060894434,95.15548897621329,103.91086738998598,106.92403611582472,115.76216727231979,125.72534759956525,128.95688953061537,140.84273174274335,151.13069543119798,163.87399183389545,171.8032189173357,186.89216628021853,200.19270470457505,219.32987417488016,236.3947885046377};
    //Input file

    TFile* infile = new TFile(infilename);
    TTree* intree = (TTree*)infile->Get("LDMX_Events");
    
    //Float_t ecalBackEnergy;
    //intree->SetBranchAddress("EcalVeto_recon.ecalBackEnergy_", &ecalBackEnergy);

    //Float_t yStd;
    //intree->SetBranchAddress("EcalVeto_recon.yStd_", &yStd);
    //TPython::Bind(yStd,"yStd");

    //Float_t xStd;
    //intree->SetBranchAddress("EcalVeto_recon.xStd_", &xStd);
    //TPython::Bind(xStd,"xStd");

    //Float_t showerRMS;
    //intree->SetBranchAddress("EcalVeto_recon.showerRMS_", &showerRMS);
    //TPython::Bind(showerRMS,"showerRMS");

    //Float_t maxCellDep;
    //intree->SetBranchAddress("EcalVeto_recon.maxCellDep_", &maxCellDep);
    //TPython::Bind(maxCellDep,"maxCellDep");
    
    //Float_t summedTightIso;
    //intree->SetBranchAddress("EcalVeto_recon.summedTightIso_", &summedTightIso);
    //TPython::Bind(summedTightIso, "summedTightIso");

    //Float_t summedDet;
    //intree->SetBranchAddress("EcalVeto_recon.summedDet_", &summedDet);
    //TPython::Bind(summedDet, "summedDet");

    //Int_t nHits;
    //intree->SetBranchAddress("EcalVeto_recon.nReadoutHits_",&nHits); 
    //TPython::Bind(nHits,"nHits");

    //Float_t stdLayerHit;
    //intree->SetBranchAddress("EcalVeto_recon.stdLayerHit_",&stdLayerHit);
    //TPython::Bind(stdLayerHit,"stdLayerHit");

    //Int_t deepestLayerHit;
    //intree->SetBranchAddress("EcalVeto_recon.deepestLayerHit_",&deepestLayerHit);
    //TPython::Bind(deepestLayerHit,"deppestLayerHit");

    //Float_t avgLayerHit;
    //intree->SetBranchAddress("EcalVeto_recon.avgLayerHit_",&avgLayerHit);
    //TPython::Bind(avgLayerHit,"avgLayerHit");

    TClonesArray* ecalHits = new TClonesArray("ldmx::EcalHit");
    intree->SetBranchAddress("ecalDigis_recon",&ecalHits);

    TClonesArray* simParticles = new TClonesArray("ldmx::SimParticle");
    intree->SetBranchAddress("SimParticles_sim", &simParticles);

    TClonesArray* simTrackerHits = new TClonesArray("ldmx::SimTrackerHit");
    intree->SetBranchAddress("EcalScoringPlaneHits_sim", &simTrackerHits);

    TClonesArray* targetSPHits = new TClonesArray("ldmx::SimTrackerHit");
    intree->SetBranchAddress("TargetScoringPlaneHits_sim",&targetSPHits);

    TClonesArray* ecalVetoResult = new TClonesArray("ldmx::EcalVetoResult");
    intree->SetBranchAddress("EcalVeto_recon",&ecalVetoResult);

    TPython::Exec("ecalVetoRes = ROOT.TClonesArray(\"ldmx::EcalVetoResult\")");
    TPython::Bind(intree,"intree");
    TPython::Exec("intree.SetBranchAddress(\"EcalVeto_recon\",ecalVetoRes)");

    //Output

    TFile* outfile = new TFile(outfilename, "RECREATE");
    TTree* outtree = new TTree("BDTree", "BDT Scores");
   
    //Is float enough? I don't think so 
    //Double_t score;
    Float_t score;
    auto branch = outtree->Branch("Score", &score);

    //Process all events

    std::cout << "Processing " << intree->GetEntriesFast() << " events" << std::endl;

    TPython::Exec("i=0");
    for(int i=0; i<intree->GetEntriesFast(); i++){
       
        //std::cout << i << std::endl;
 
        intree->GetEntry(i);

        //Find electron and photon trajectories
        std::vector<double> pvec;
        std::vector<float> pos;
        bool foundElectron = false;
        bool foundTargetElectron = false;
        //Electron at target
        std::vector<double> pvec0;
        std::vector<float> pos0;

        double pz;

        for(int k=0; k < simParticles->GetEntriesFast(); k++){
            ldmx::SimParticle* particle = (ldmx::SimParticle*)simParticles->At(k);
            if(particle->getPdgID() == 11 && particle->getParentCount() == 0){

                int electronTrackID = particle->getTrackID();

                for(int j=0; j < simTrackerHits->GetEntriesFast(); j++){
                    
                    ldmx::SimTrackerHit* hit = (ldmx::SimTrackerHit*)simTrackerHits->At(j);

                    //Doesn't work? Why?
                    //auto e = hit->getSimParticle();
                    //Match the TrackID instead
                    int hitTrackID = hit->getTrackID();

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

                
        //if(foundElectron == false){
            //std::cout << "Could not find electron hit on ECal in event: " << i << std::endl;
            //std::cout << simTrackerHits->GetEntriesFast() << std::endl;
            //std::cout << pz << std::endl;
            //continue;
        //}
        
        //if(foundTargetElectron == false){
            //std::cout << "Could not find electron hit on target in event: " << i << std::endl;

            //continue;
        //}
           
        
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
            //std::vector<double> photonP{-pvec0.at(0),-pvec0.at(1),pvec0.at(2)};
            photonP.push_back(-pvec0.at(0));
            photonP.push_back(-pvec0.at(1));
            photonP.push_back(8000-pvec0.at(2));
            for(auto ite = layerZs.begin(); ite < layerZs.end(); ite++){
                photonPosX.push_back(pos0.at(0) + photonP.at(0)/photonP.at(2)*(*ite - pos0.at(2)));
                photonPosY.push_back(pos0.at(1) + photonP.at(1)/photonP.at(2)*(*ite - pos0.at(2)));
            }
        }

        //std::cout << "PhotonP " << photonP.at(0) << ":" << photonP.at(1) << ":" << photonP.at(2) << std::endl;
       
        double theta = -1;
        double magnitude = -1;
 

        if(foundElectron){ 
            theta = TMath::ACos(pvec.at(2)/TMath::Sqrt(pow(pvec.at(0),2)+pow(pvec.at(1),2)+pow(pvec.at(2),2)));

            magnitude = TMath::Sqrt(pow(pvec.at(0),2)+pow(pvec.at(1),2)+pow(pvec.at(2),2));
        }
        
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
        
        double ele68ContEnergy = 0;
        double ele68x2ContEnergy = 0;
        double ele68x3ContEnergy = 0;
        double ele68x4ContEnergy = 0;
        double ele68x5ContEnergy = 0;

        double photon68ContEnergy = 0;
        double photon68x2ContEnergy = 0;
        double photon68x3ContEnergy = 0;
        double photon68x4ContEnergy = 0;
        double photon68x5ContEnergy = 0;

        double overlap68ContEnergy = 0;
        double overlap68x2ContEnergy = 0;
        double overlap68x3ContEnergy = 0;
        double overlap68x4ContEnergy = 0;
        double overlap68x5ContEnergy = 0;

        double outside68ContEnergy = 0;
        double outside68x2ContEnergy = 0;
        double outside68x3ContEnergy = 0;
        double outside68x4ContEnergy = 0;
        double outside68x5ContEnergy = 0;

        double outside68ContNHits = 0;
        double outside68x2ContNHits = 0;
        double outside68x3ContNHits = 0;
        double outside68x4ContNHits = 0;
        double outside68x5ContNHits = 0;

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
        

        std::vector<std::vector<double>> outside68HitPositions;
        std::vector<std::vector<double>> outside68x2HitPositions;
        std::vector<std::vector<double>> outside68x3HitPositions;
        std::vector<std::vector<double>> outside68x4HitPositions;
        std::vector<std::vector<double>> outside68x5HitPositions;

        double ecalBackEnergy = 0;

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

                double distancePhoton = TMath::Sqrt(pow(hitX -photonPosX.at(layer),2)+pow(hitY-photonPosY.at(layer),2));

                double hitE = hit->getEnergy();
               
                if(!(hitE > 0)){
                    continue;
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
                
                if(distancePhoton < ip_radius){
                    photon68ContEnergy += hitE;
                    photon68Totals.at(layer) += hitE;
                }
                else if(distancePhoton < 2*ip_radius){
                    photon68x2ContEnergy += hitE;
                }
                else if(distancePhoton < 3*ip_radius){
                    photon68x3ContEnergy += hitE;
                }
                else if(distancePhoton < 4*ip_radius){
                    photon68x4ContEnergy += hitE;
                }
                else if(distancePhoton < 5*ip_radius){
                    photon68x5ContEnergy += hitE;
                }

                if(distanceEle < ir_radius && distancePhoton < ip_radius){
                    overlap68ContEnergy += hitE;
                    overlap68Totals.at(layer) += hitE;
                }
                if(distanceEle < 2*ir_radius && distancePhoton < 2*ip_radius){
                    overlap68x2ContEnergy += hitE;
                }
                if(distanceEle < 3*ir_radius && distancePhoton < 3*ip_radius){
                    overlap68x3ContEnergy += hitE;
                }
                if(distanceEle < 4*ir_radius && distancePhoton < 4*ip_radius){
                    overlap68x4ContEnergy += hitE;
                }
                if(distanceEle < 5*ir_radius && distancePhoton < 5*ip_radius){
                    overlap68x5ContEnergy += hitE;
                }

                std::vector<double> hitPosition{hitX,hitY,(Float_t)layer,hitE};

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
                    outside68HitPositions.push_back(hitPosition);           
                }
                if(distanceEle > 2*ir_radius && distancePhoton > 2*ip_radius){
                    outside68x2ContEnergy += hitE;
                    outside68x2ContNHits += 1;
                    outside68x2ContXmean += hitX*hitE;
                    outside68x2ContYmean += hitY*hitE;
                    outside68x2WgtCentroidCoordsX += hitX*hitE;
                    outside68x2WgtCentroidCoordsY += hitY*hitE;
                    outside68x2HitPositions.push_back(hitPosition);           
                }
                if(distanceEle > 3*ir_radius && distancePhoton > 3*ip_radius){
                    outside68x3ContEnergy += hitE;
                    outside68x3ContNHits += 1;
                    outside68x3ContXmean += hitX*hitE;
                    outside68x3ContYmean += hitY*hitE;
                    outside68x3WgtCentroidCoordsX += hitX*hitE;
                    outside68x3WgtCentroidCoordsY += hitY*hitE;
                    outside68x3HitPositions.push_back(hitPosition);           
                }
                if(distanceEle > 4*ir_radius && distancePhoton > 4*ip_radius){
                    outside68x4ContEnergy += hitE;
                    outside68x4ContNHits += 1;
                    outside68x4ContXmean += hitX*hitE;
                    outside68x4ContYmean += hitY*hitE;
                    outside68x4WgtCentroidCoordsX += hitX*hitE;
                    outside68x4WgtCentroidCoordsY += hitY*hitE;
                    outside68x4HitPositions.push_back(hitPosition);           
                }
                if(distanceEle > 5*ir_radius && distancePhoton > 5*ip_radius){
                    outside68x5ContEnergy += hitE;
                    outside68x5ContNHits += 1;
                    outside68x5ContXmean += hitX*hitE;
                    outside68x5ContYmean += hitY*hitE;
                    outside68x5WgtCentroidCoordsX += hitX*hitE;
                    outside68x5WgtCentroidCoordsY += hitY*hitE;
                    outside68x5HitPositions.push_back(hitPosition);           
                }
            }

        }

        double outside68ContShowerRMS = 0;
        double outside68ContXstd = 0;       
        double outside68ContYstd = 0;       
        std::vector<double> outside68Xstd{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        std::vector<double> outside68Ystd{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

        for(auto hit = outside68HitPositions.begin(); hit < outside68HitPositions.end(); hit++){
            double distanceCentroid = TMath::Sqrt(pow(hit->at(0)-outside68WgtCentroidCoordsX,2)+pow(hit->at(1)-outside68WgtCentroidCoordsY,2));

            outside68ContShowerRMS += distanceCentroid*(hit->at(3));

            //Converted back from float, hope it's okay
            int layer = (int)hit->at(2);
            if(outside68Totals.at(layer) > 0){
                outside68Xstd.at(layer) += pow(hit->at(0)-outside68Xmean.at(layer)/outside68Totals.at(layer),2)*(hit->at(3));
                outside68Ystd.at(layer) += pow(hit->at(1)-outside68Ymean.at(layer)/outside68Totals.at(layer),2)*(hit->at(3));
            }
            if(outside68ContEnergy > 0){
                outside68ContXstd += pow(hit->at(0)-outside68ContXmean/outside68ContEnergy,2)*(hit->at(3));
                outside68ContYstd += pow(hit->at(1)-outside68ContYmean/outside68ContEnergy,2)*(hit->at(3));
            }

        }
        
        double outside68x2ContShowerRMS = 0;
        double outside68x2ContXstd = 0;       
        double outside68x2ContYstd = 0;
        
        for(auto hit = outside68x2HitPositions.begin(); hit < outside68x2HitPositions.end(); hit++){
            double distanceCentroid = TMath::Sqrt(pow(hit->at(0)-outside68x2WgtCentroidCoordsX,2)+pow(hit->at(1)-outside68x2WgtCentroidCoordsY,2));

            outside68x2ContShowerRMS += distanceCentroid*(hit->at(3));

            if(outside68x2ContEnergy > 0){
                outside68x2ContXstd += pow(hit->at(0)-outside68x2ContXmean/outside68x2ContEnergy,2)*(hit->at(3));
                outside68x2ContYstd += pow(hit->at(1)-outside68x2ContYmean/outside68x2ContEnergy,2)*(hit->at(3));
            }
        }
        
        double outside68x3ContShowerRMS = 0;
        double outside68x3ContXstd = 0;       
        double outside68x3ContYstd = 0;
        
        for(auto hit = outside68x3HitPositions.begin(); hit < outside68x3HitPositions.end(); hit++){
            double distanceCentroid = TMath::Sqrt(pow(hit->at(0)-outside68x3WgtCentroidCoordsX,2)+pow(hit->at(1)-outside68x3WgtCentroidCoordsY,2));

            outside68x3ContShowerRMS += distanceCentroid*(hit->at(3));

            if(outside68x3ContEnergy > 0){
                outside68x3ContXstd += pow(hit->at(0)-outside68x3ContXmean/outside68x3ContEnergy,2)*(hit->at(3));
                outside68x3ContYstd += pow(hit->at(1)-outside68x3ContYmean/outside68x3ContEnergy,2)*(hit->at(3));
            }
        }
        
        double outside68x4ContShowerRMS = 0;
        double outside68x4ContXstd = 0;       
        double outside68x4ContYstd = 0;
        
        for(auto hit = outside68x4HitPositions.begin(); hit < outside68x4HitPositions.end(); hit++){
            double distanceCentroid = TMath::Sqrt(pow(hit->at(0)-outside68x4WgtCentroidCoordsX,2)+pow(hit->at(1)-outside68x4WgtCentroidCoordsY,2));

            outside68x4ContShowerRMS += distanceCentroid*(hit->at(3));

            if(outside68x4ContEnergy > 0){
                outside68x4ContXstd += pow(hit->at(0)-outside68x4ContXmean/outside68x4ContEnergy,2)*(hit->at(3));
                outside68x4ContYstd += pow(hit->at(1)-outside68x4ContYmean/outside68x4ContEnergy,2)*(hit->at(3));
            }
        }
        
        double outside68x5ContShowerRMS = 0;
        double outside68x5ContXstd = 0;       
        double outside68x5ContYstd = 0;
        
        for(auto hit = outside68x5HitPositions.begin(); hit < outside68x5HitPositions.end(); hit++){
            double distanceCentroid = TMath::Sqrt(pow(hit->at(0)-outside68x5WgtCentroidCoordsX,2)+pow(hit->at(1)-outside68x5WgtCentroidCoordsY,2));

            outside68x5ContShowerRMS += distanceCentroid*(hit->at(3));

            if(outside68x5ContEnergy > 0){
                outside68x5ContXstd += pow(hit->at(0)-outside68x5ContXmean/outside68x5ContEnergy,2)*(hit->at(3));
                outside68x5ContYstd += pow(hit->at(1)-outside68x5ContYmean/outside68x5ContEnergy,2)*(hit->at(3));
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
 
        //Evaluate BDT score
        TPython::Exec("intree.GetEntry(i); i +=1");

        Int_t nHits = TPython::Eval("ecalVetoRes[0].getNReadoutHits()");
        Float_t summedDet = TPython::Eval("ecalVetoRes[0].getSummedDet()");
        Float_t summedTightIso = TPython::Eval("ecalVetoRes[0].getSummedTightIso()");
        Float_t maxCellDep = TPython::Eval("ecalVetoRes[0].getMaxCellDep()");
        Float_t showerRMS = TPython::Eval("ecalVetoRes[0].getShowerRMS()");
        Float_t xStd = TPython::Eval("ecalVetoRes[0].getXStd()");
        Float_t yStd = TPython::Eval("ecalVetoRes[0].getYStd()");
        Float_t avgLayerHit = TPython::Eval("ecalVetoRes[0].getAvgLayerHit()");
        Float_t stdLayerHit = TPython::Eval("ecalVetoRes[0].getStdLayerHit()");
        Int_t deepestLayerHit = TPython::Eval("ecalVetoRes[0].getDeepestLayerHit()");

        score = -1.0;
               
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

        outtree->Fill();
    }

    outtree->Write();
    outfile->Close();
}

