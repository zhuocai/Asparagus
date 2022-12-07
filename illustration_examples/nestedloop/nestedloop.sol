pragma solidity >=0.4.0 <= 0.4.26;

contract NestedLoop{
    
    function main(uint a, uint b) public returns(uint){
        uint count = 0;
        for (uint i = 0; i < a; ++i) {
            for (uint j = 0; j < b; ++j) {
                ++count;
            }
        }
        return count;
    }   

}
