// pragma solidity ^0.8.12;
pragma solidity ^0.4.16;

/// @title Voting with delegation.
contract Voting {
    // This declares a new complex type which will
    // be used for variables later.
    // It will represent a single voter.
    

    // This is a type for a single proposal.
    struct Proposal {
        bytes32 name;   // short name (up to 32 bytes)
        uint voteCount; // number of accumulated votes
    }
    // A dynamically-sized array of `Proposal` structs.
    Proposal[] public proposals;

   
    function winningProposal() public view
            returns (uint winningProposal)
    {
        uint winningVoteCount = 0;
        
        for (uint p = 0; p < proposals.length; p++) {
            if (proposals[p].voteCount > winningVoteCount) {
                winningVoteCount = proposals[p].voteCount;
                winningProposal = p;
            }
        }
    }
}
