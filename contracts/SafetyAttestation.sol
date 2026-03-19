// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title SafetyAttestation — On-chain safety audit attestation for multi-agent systems
/// @notice Stores deterministic hashes of SwarmGym safety audit reports on Base.
///         Full metrics live off-chain; this contract anchors the hash for verifiability.
/// @dev Designed for the Synthesis hackathon (ERC-8004 compatible agent identities).
contract SafetyAttestation {

    // -----------------------------------------------------------------------
    // Types
    // -----------------------------------------------------------------------

    struct Attestation {
        bytes32 metricsHash;      // SHA-256 of canonical metrics JSON
        string  agentId;          // Agent identifier (off-chain or ERC-8004 address)
        string  safetyGrade;      // A/B/C/D/F
        bool    adverseSelection; // True if quality_gap < 0
        uint256 interactionCount; // Number of interactions audited
        uint256 timestamp;        // Block timestamp at submission
        address submitter;        // Address that submitted the attestation
    }

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    /// @notice All attestations in submission order
    Attestation[] public attestations;

    /// @notice Agent ID -> list of attestation indices
    mapping(bytes32 => uint256[]) private _agentAttestations;

    /// @notice Metrics hash -> attestation index (uniqueness check)
    mapping(bytes32 => uint256) private _hashIndex;
    mapping(bytes32 => bool) private _hashExists;

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------

    event AttestationSubmitted(
        uint256 indexed attestationId,
        bytes32 indexed metricsHash,
        string  agentId,
        string  safetyGrade,
        address submitter
    );

    // -----------------------------------------------------------------------
    // Errors
    // -----------------------------------------------------------------------

    error DuplicateHash(bytes32 metricsHash);
    error EmptyAgentId();
    error EmptyGrade();

    // -----------------------------------------------------------------------
    // Write
    // -----------------------------------------------------------------------

    /// @notice Submit a new safety attestation
    /// @param metricsHash SHA-256 hash of the canonical metrics JSON
    /// @param agentId Agent identifier string
    /// @param safetyGrade Letter grade (A-F)
    /// @param adverseSelection Whether adverse selection was detected
    /// @param interactionCount Number of interactions in the audit
    /// @return attestationId Index of the new attestation
    function submitAttestation(
        bytes32 metricsHash,
        string calldata agentId,
        string calldata safetyGrade,
        bool adverseSelection,
        uint256 interactionCount
    ) external returns (uint256 attestationId) {
        if (bytes(agentId).length == 0) revert EmptyAgentId();
        if (bytes(safetyGrade).length == 0) revert EmptyGrade();
        if (_hashExists[metricsHash]) revert DuplicateHash(metricsHash);

        attestationId = attestations.length;
        attestations.push(Attestation({
            metricsHash: metricsHash,
            agentId: agentId,
            safetyGrade: safetyGrade,
            adverseSelection: adverseSelection,
            interactionCount: interactionCount,
            timestamp: block.timestamp,
            submitter: msg.sender
        }));

        bytes32 agentKey = keccak256(bytes(agentId));
        _agentAttestations[agentKey].push(attestationId);
        _hashIndex[metricsHash] = attestationId;
        _hashExists[metricsHash] = true;

        emit AttestationSubmitted(
            attestationId,
            metricsHash,
            agentId,
            safetyGrade,
            msg.sender
        );
    }

    // -----------------------------------------------------------------------
    // Read
    // -----------------------------------------------------------------------

    /// @notice Total number of attestations
    function attestationCount() external view returns (uint256) {
        return attestations.length;
    }

    /// @notice Get attestation by index
    function getAttestation(uint256 id) external view returns (Attestation memory) {
        return attestations[id];
    }

    /// @notice Check if a metrics hash has been attested
    function isAttested(bytes32 metricsHash) external view returns (bool) {
        return _hashExists[metricsHash];
    }

    /// @notice Get attestation ID for a metrics hash
    function getByHash(bytes32 metricsHash) external view returns (Attestation memory) {
        require(_hashExists[metricsHash], "Hash not found");
        return attestations[_hashIndex[metricsHash]];
    }

    /// @notice Get all attestation IDs for an agent
    function getAgentAttestations(string calldata agentId)
        external view returns (uint256[] memory)
    {
        return _agentAttestations[keccak256(bytes(agentId))];
    }

    /// @notice Verify that a given metrics hash matches an attestation
    function verify(bytes32 metricsHash, string calldata agentId)
        external view returns (bool verified, uint256 attestationId)
    {
        if (!_hashExists[metricsHash]) return (false, 0);
        uint256 idx = _hashIndex[metricsHash];
        Attestation storage att = attestations[idx];
        verified = keccak256(bytes(att.agentId)) == keccak256(bytes(agentId));
        attestationId = idx;
    }
}
