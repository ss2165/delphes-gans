void trim_macro(char* fil){
    gSystem->Load("~/MG5_aMC_v2_2_3/Delphes/libDelphes");
    TChain chain("Delphes");
    chain.Add(fil);
    ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);
    Long64_t numberOfEntries = treeReader->GetEntries();
    TClonesArray *branchJet = treeReader->UseBranch("Jet");

    for(Int_t entry = 0; entry<numberOfEntries; ++entry){
        treeReader->ReadEntry(entry);
        if(branchJet->GetEntries()>0) {
            Jet *jet = (Jet*) branchJet->At(0); //only take leading jet
            cout<<entry<<",";
            for(int i = 0; i<5; i++){
                for(int j = 0; j<4; j++){
                    cout<<jet->TrimmedP4[i][j]<<",";
                }
            }

        }
    }
    cout<<endl;
}